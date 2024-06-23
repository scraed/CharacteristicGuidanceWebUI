import modules.scripts as scripts
import gradio as gr

import io
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import inspect
import torch
from modules import prompt_parser, devices, sd_samplers_common
import re
from modules.shared import opts, state
import modules.shared as shared
from modules.script_callbacks import CFGDenoiserParams, cfg_denoiser_callback
from modules.script_callbacks import CFGDenoisedParams, cfg_denoised_callback
from modules.script_callbacks import AfterCFGCallbackParams, cfg_after_cfg_callback

from k_diffusion.external import CompVisVDenoiser, CompVisDenoiser
from modules.sd_samplers_timesteps import CompVisTimestepsDenoiser, CompVisTimestepsVDenoiser
from modules.sd_samplers_cfg_denoiser import  catenate_conds, subscript_cond, pad_cond
from modules import script_callbacks
from modules_forge import forge_sampler
import k_diffusion.utils as utils_old
from ldm_patched.modules.conds import CONDRegular, CONDCrossAttn
from ldm_patched.modules import model_management
from ldm_patched.modules.ops import cleanup_cache
from ldm_patched.modules.samplers import *
from modules_forge.forge_sampler import *



# 1st edit by https://github.com/comfyanonymous/ComfyUI
# 2nd edit by Forge Official

def solve_least_squares(A, B):
    # print(A.shape)
    # print(B.shape)
    # Compute C = A^T A
    # min_eigenvalues = torch.min( torch.linalg.eigvalsh(C), dim=-1 )
    # eps_e = torch.maximum( min_eigenvalues, min_eigenvalues.new_ones(min_eigenvalues.shape)*1e-3 )[...,]
    C = torch.matmul(A.transpose(-2, -1), A)  # + eps_e*torch.eye(A.shape[-1], device=A.device)
    # Compute the pseudo-inverse of C
    U, S, Vh = torch.linalg.svd(C.float(), full_matrices=False)
    D_inv = torch.diag_embed(1.0 / torch.maximum(S, torch.ones_like(S) * 1e-4))
    C_inv = Vh.transpose(-1,-2).matmul(D_inv).matmul(U.transpose(-1,-2))

    # Compute X = C_inv A^T B
    X = torch.matmul(torch.matmul(C_inv, A.transpose(-2, -1)), B)
    return X


def split_basis(g, n):
    # Define the number of quantiles, n

    # Flatten the last two dimensions of g for easier processing
    g_flat = g.view(g.shape[0], g.shape[1], -1)  # Shape will be (6, 4, 64*64)

    # Calculate quantiles
    quantiles = torch.quantile(g_flat, torch.linspace(0, 1, n + 1, device=g.device), dim=-1).permute(1, 2, 0)

    # Initialize an empty tensor for the output
    output = torch.zeros(*g.shape, n, device=g.device)

    # Use broadcasting and comparisons to fill the output tensor
    for i in range(n):
        lower = quantiles[..., i][..., None, None]
        upper = quantiles[..., i + 1][..., None, None]
        if i < n - 1:
            mask = (g >= lower) & (g < upper)
        else:
            mask = (g >= lower) & (g <= upper)
        output[..., i] = g * mask

    # Reshape output to the desired shape
    output = output.view(*g.shape, n)
    return output

def proj_least_squares(A, B, reg):
    # print(A.shape)
    # print(B.shape)
    # Compute C = A^T A
    C = torch.matmul(A.transpose(-2, -1), A) + reg * torch.eye(A.shape[-1], device=A.device)

    # Compute the eigenvalues and eigenvectors of C
    eigenvalues, eigenvectors = torch.linalg.eigh(C)
    # eigenvalues = torch.maximum( eigenvalues,eigenvalues*0+1e-3  )

    # Diagonal matrix with non-zero eigenvalues in the diagonal
    D_inv = torch.diag_embed(1.0 / torch.maximum(eigenvalues, torch.ones_like(eigenvalues) * 1e-4))

    # Compute the pseudo-inverse of C
    C_inv = torch.matmul(torch.matmul(eigenvectors, D_inv), eigenvectors.transpose(-2, -1))

    # Compute X = C_inv A^T B
    B_proj = torch.matmul(A, torch.matmul(torch.matmul(C_inv, A.transpose(-2, -1)), B))
    return B_proj

print("**********Read forge sample code *********")

def calc_cond_uncond_batch(self,model, cond, uncond, x_in, timestep, model_options,cond_scale):
    def Chara_iteration(self,model,dxs,x_in, sigma_in,cond_scale,uncond, c):
        #######################################
        def evaluation(func, x_in, t_in, c_concat=None, c_crossattn=None, control=None, transformer_options={}, *args, **kwargs):
            # tensor, uncond, cond_in = conds
            # print('x_in eval',x_in.shape)
            return func(x_in, t_in, c_concat, c_crossattn, control, transformer_options, *args, **kwargs)

        def eps_evaluation(x_in, t_in, c_concat=None, c_crossattn=None, control=None, transformer_options={}):
            # print('c concat',c_concat.shape)
            # print('c_crossattn 0',(c_crossattn)[0][:10])
            # print('c_crossattn 1', (c_crossattn)[1][:10])
            print('model eps evaluation')
            x_out = model.apply_model(x_in,t_in, c_concat, c_crossattn, control, transformer_options)
            t_in_expand = t_in.view(t_in.shape[:1] + (1,) * (x_in.ndim - 1))
            eps_out = (x_in - x_out)/t_in_expand
            return eps_out

        def v_evaluation(x_in, t_in, c_concat=None, c_crossattn=None, control=None, transformer_options={}):
            print('model v evaluation')
            x_out = model.apply_model(x_in, t_in, c_concat, c_crossattn, control, transformer_options)
            t_in_expand = t_in.view(t_in.shape[:1] + (1,) * (x_in.ndim - 1))
            sigma_data = model.model_sampling.sigma_data
            v_out = (x_in* sigma_data**2 - (sigma_data**2 + t_in_expand**2)*x_out)/(t_in_expand*sigma_data*(t_in_expand**2+sigma_data**2)** 0.5)
            return v_out

        def x_out_evaluation(x_in, t_in, c_concat=None, c_crossattn=None, control=None, transformer_options={}):
            # t_in_expand = t_in.view(t_in.shape[:1] + (1,) * (x_in.ndim - 1))
            # x_in =  x_in*((t_in_expand ** 2 + 1 ** 2) ** 0.5)
            x_out = model.apply_model(x_in, t_in, c_concat, c_crossattn, control, transformer_options)
            return x_out
            # return self.inner_model.get_eps(x_in, t_in, cond=make_condition_dict(cond_in, image_cond_in))

        # def x_out_evaluation(x_in, cond_in, sigma_in, image_cond_in):
        #     return self.inner_model(x_in, sigma_in, cond=make_condition_dict(cond_in, image_cond_in))
        if isinstance(self.inner_model, CompVisDenoiser):
            t_in = self.inner_model.sigma_to_t(sigma_in)
            # print('t_in (sigma to t)', t_in)
            abt = self.inner_model.inner_model.alphas_cumprod.to(t_in.device)[t_in.long()]
            # print('abt', abt)
            c_out, c_in = [utils_old.append_dims(x, x_in.ndim) for x in self.inner_model.get_scalings(sigma_in)]
            # print('c_in', c_in)
        elif isinstance(self.inner_model, CompVisVDenoiser):
            t_in = self.inner_model.sigma_to_t(sigma_in)
            abt = self.inner_model.inner_model.alphas_cumprod.to(t_in.device)[t_in.long()]
            c_skip, c_out, c_in = [utils_old.append_dims(x, x_in.ndim) for x in self.inner_model.get_scalings(sigma_in)]
        elif isinstance(self.inner_model, CompVisTimestepsDenoiser) or isinstance(self.inner_model,
                                                                                  CompVisTimestepsVDenoiser):
            t_in = sigma_in
            abt = self.alphas[t_in.long()]
        else:
            raise NotImplementedError()
        # t_in = self.inner_model.sigma_to_t(sigma_in)
        # # print('t_in',t_in.device)
        # # print('acd',self.inner_model.inner_model.alphas_cumprod.device)
        # abt = self.inner_model.inner_model.alphas_cumprod.to(t_in.device)[t_in.long()]
        # print('abt',abt)
        # c_out, c_in = [utils_old.append_dims(x, x_in.ndim) for x in self.inner_model.get_scalings(sigma_in)]
        # print('c in',c_in)
        # x_real = x_in#model.model_sampling.calculate_input(sigma_in, x_in)
        # dxs = None
        if dxs is None:
            dxs = torch.zeros_like(x_in[-uncond.shape[0]:])
        res_thres = self.res_thres
        num_x_in_cond = len(x_in[:-uncond.shape[0]]) // len(dxs)

        h = cond_scale * num_x_in_cond

        scale = ((1 - abt) ** 0.5)[-uncond.shape[0]:, None, None, None]
        abt_current = abt[-uncond.shape[0]:, None, None, None]
        abt_smallest = self.inner_model.inner_model.alphas_cumprod[-1]

        dxs_Anderson = []
        g_Anderson = []

        def AndersonAccR(dxs, g, reg_level, reg_target, pre_condition=None, m=3):
            batch = dxs.shape[0]
            x_shape = dxs.shape[1:]
            reg_residual_form = reg_level
            g_flat = g.reshape(batch, -1)
            dxs_flat = dxs.reshape(batch, -1)
            res_g = self.reg_size * (reg_residual_form[:, None] - reg_target[:, None])
            res_dxs = reg_residual_form[:, None]
            g_Anderson.append(torch.cat((g_flat, res_g), dim=-1))
            dxs_Anderson.append(torch.cat((dxs_flat, res_dxs), dim=-1))

            if len(g_Anderson) < 2:
                return dxs, g, res_dxs[:, 0], res_g[:, 0]
            else:
                g_Anderson[-2] = g_Anderson[-1] - g_Anderson[-2]
                dxs_Anderson[-2] = dxs_Anderson[-1] - dxs_Anderson[-2]
                if len(g_Anderson) > m:
                    del dxs_Anderson[0]
                    del g_Anderson[0]
                gA = torch.cat([g[..., None] for g in g_Anderson[:-1]], dim=-1)
                gB = g_Anderson[-1][..., None]

                gA_norm = torch.maximum(torch.sum(gA ** 2, dim=-2, keepdim=True) ** 0.5, torch.ones_like(gA) * 1e-4)
                # print("gA_norm ",gA_norm.shape)
                # gB_norm = torch.sum( gB**2, dim = -2 , keepdim=True )**0.5 + 1e-6
                # gamma = solve_least_squares(gA/gA_norm, gB)
                gamma = torch.linalg.lstsq(gA / gA_norm, gB).solution
                if torch.sum(torch.isnan(gamma)) > 0:
                    gamma = solve_least_squares(gA / gA_norm, gB)
                xA = torch.cat([x[..., None] for x in dxs_Anderson[:-1]], dim=-1)
                xB = dxs_Anderson[-1][..., None]
                # print("xO print",xB.shape, xA.shape, gA_norm.shape, gamma.shape)
                xO = xB - (xA / gA_norm).matmul(gamma)
                gO = gB - (gA / gA_norm).matmul(gamma)
                dxsO = xO[:, :-1].reshape(batch, *x_shape)
                dgO = gO[:, :-1].reshape(batch, *x_shape)
                resxO = xO[:, -1, 0]
                resgO = gO[:, -1, 0]
                # print("xO",xO.shape)
                # print("gO",gO.shape)
                # print("gamma",gamma.shape)
                return dxsO, dgO, resxO, resgO

        def downsample_reg_g(dx, g_1, reg):
            # DDec_dx = DDec(dx)
            # down_DDec_dx = downsample(DDec_dx, factor=factor)
            # DEnc_dx = DEnc(down_DDec_dx)
            # return DEnc_dx

            if g_1 is None:
                return dx
            elif self.noise_base >= 1:
                # return g_1*torch.sum(g_1*dx, dim = (-1,-2), keepdim=True )/torch.sum( g_1**2, dim = (-1,-2) , keepdim=True )
                A = g_1.reshape(g_1.shape[0] * g_1.shape[1], g_1.shape[2] * g_1.shape[3], g_1.shape[4])
                B = dx.reshape(dx.shape[0] * dx.shape[1], -1, 1)
                regl = reg[:, None].expand(-1, dx.shape[1]).reshape(dx.shape[0] * dx.shape[1], 1, 1)
                dx_proj = proj_least_squares(A, B, regl)

                return dx_proj.reshape(*dx.shape)
            else:
                # return g_1*torch.sum(g_1*dx, dim = (-1,-2), keepdim=True )/torch.sum( g_1**2, dim = (-1,-2) , keepdim=True )
                A = g_1.reshape(g_1.shape[0], g_1.shape[1] * g_1.shape[2] * g_1.shape[3], g_1.shape[4])
                B = dx.reshape(dx.shape[0], -1, 1)
                regl = reg[:, None].reshape(dx.shape[0], 1, 1)
                dx_proj = proj_least_squares(A, B, regl)

                return dx_proj.reshape(*dx.shape)

        g_1 = None

        reg_level = torch.zeros(dxs.shape[0], device=dxs.device) + max(5, self.reg_ini)
        reg_target_level = self.reg_ini * (abt_smallest / abt_current[:, 0, 0, 0]) ** (1 / self.reg_range)
        Converged = False
        eps0_ch, eps1_ch = torch.zeros_like(dxs), torch.zeros_like(dxs)
        best_res_el = torch.mean(dxs, dim=(-1, -2, -3), keepdim=True) + 100
        best_res = 100
        best_dxs = torch.zeros_like(dxs)
        res_max = torch.zeros(dxs.shape[0], device=dxs.device)
        n_iterations = self.ite
        if self.dxs_buffer is not None:
            abt_prev = self.abt_buffer
            dxs = self.dxs_buffer
            # if self.CFGdecayS:
            # print('dxs device',dxs.device)
            # print('abt_current device', abt_current.device)
            # print('abt_prev',abt_prev.device)
            dxs = dxs * ((abt_prev.to(dxs.device) - abt_current.to(dxs.device) * abt_prev.to(dxs.device)) / (abt_current.to(dxs.device) - abt_current.to(dxs.device) * abt_prev.to(dxs.device)))
            # print(abt_prev.shape, abt_current.shape, self.dxs_buffer.shape)
            dxs = self.chara_decay * dxs
        iteration_counts = 0
        for iteration in range(n_iterations):
            def compute_correction_direction(dxs):
                dxs_cond_part = torch.cat([*([(h - 1) * dxs[:, None, ...]] * num_x_in_cond)], axis=1).view(
                    (dxs.shape[0] * num_x_in_cond, *dxs.shape[1:]))
                dxs_add = torch.cat([dxs_cond_part, h * dxs], axis=0)

                if isinstance(self.inner_model, CompVisDenoiser):
                    # eps_out = self.inner_model.get_eps(x_in * c_in + dxs_add * c_in, t_in, cond=cond)
                    # eps_out = evaluation(eps_out_evaluation, x_in * c_in + dxs_add * c_in, sigma_in,
                    #                      **c)
                    eps_out = evaluation(eps_evaluation, x_in + dxs_add, sigma_in,
                                         **c)
                    # print('eps out',eps_out.shape)
                    # print('uncond',-uncond.shape[0])
                    # pred_eps_uncond = eps_out[-uncond.shape[0]:]
                    # eps_cond_batch = eps_out[:-uncond.shape[0]]
                    pred_eps_uncond = eps_out[:-uncond.shape[0]] # forge: c_crossatten[0]: uncondition
                    eps_cond_batch = eps_out[-uncond.shape[0]:] # forge: c_crossatten[1]: condition
                    eps_cond_batch_target_shape = (
                    len(eps_cond_batch) // num_x_in_cond, num_x_in_cond, *(eps_cond_batch.shape[1:]))
                    pred_eps_cond = torch.mean(eps_cond_batch.view(eps_cond_batch_target_shape), dim=1, keepdim=False)
                    # print('pred_eps_uncond device',pred_eps_uncond.device)
                    # print('pred_eps_cond',pred_eps_cond.device)
                    # print('scale',scale)
                    ggg = (pred_eps_uncond - pred_eps_cond) * scale.to(c_in.device) / c_in[-uncond.shape[0]:]
                elif isinstance(self.inner_model, CompVisVDenoiser):
                    # v_out = self.inner_model.get_v(x_in * c_in + dxs_add * c_in, t_in, cond=cond)
                    v_out = evaluation(v_evaluation, x_in+dxs_add,sigma_in, **c)
                    eps_out = -c_out * x_in + c_skip ** 0.5 * v_out
                    # pred_eps_uncond = eps_out[-uncond.shape[0]:]
                    # eps_cond_batch = eps_out[:-uncond.shape[0]]
                    pred_eps_uncond = eps_out[:-uncond.shape[0]] # forge: c_crossatten[0]: uncondition
                    eps_cond_batch = eps_out[-uncond.shape[0]:] # forge: c_crossatten[1]: condition
                    eps_cond_batch_target_shape = (
                    len(eps_cond_batch) // num_x_in_cond, num_x_in_cond, *(eps_cond_batch.shape[1:]))
                    pred_eps_cond = torch.mean(eps_cond_batch.view(eps_cond_batch_target_shape), dim=1, keepdim=False)
                    ggg = (pred_eps_uncond - pred_eps_cond) * scale / c_in[-uncond.shape[0]:]
                # elif isinstance(self.inner_model, CompVisTimestepsDenoiser) or isinstance(self.inner_model,
                #                                                                           CompVisTimestepsVDenoiser):
                #     # eps_out = self.inner_model(x_in + dxs_add, t_in, cond=cond)
                #     eps_out = evaluation(eps_legacy_evaluation, x_in + dxs_add, (tensor, uncond, cond_in), t_in,
                #                          image_cond_in)
                #     pred_eps_uncond = eps_out[-uncond.shape[0]:]
                #     eps_cond_batch = eps_out[:-uncond.shape[0]]
                #     eps_cond_batch_target_shape = (
                #     len(eps_cond_batch) // num_x_in_cond, num_x_in_cond, *(eps_cond_batch.shape[1:]))
                #     pred_eps_cond = torch.mean(eps_cond_batch.view(eps_cond_batch_target_shape), dim=1, keepdim=False)
                #     ggg = (pred_eps_uncond - pred_eps_cond) * scale
                else:
                    raise NotImplementedError()
                return ggg
            ggg = compute_correction_direction(dxs)
            g = dxs - downsample_reg_g(ggg, g_1, reg_level)
            if g_1 is None:
                g_basis = -compute_correction_direction(dxs * 0)
                g_1 = split_basis(g_basis, max(self.noise_base, 1))
                # if self.Projg:
                #        g_1 = split_basis( g, self.noise_base)
                # else:
                #        g_1 = split_basis( ggg, self.noise_base)
                # if self.CFGdecayS and self.dxs_buffer is not None:
                #     g_1 = torch.cat( [g_1, self.dxs_buffer[:,:,:,:,None]], dim=-1 )
                # if self.noise_base > 0:
                #    noise_base = torch.randn(g_1.shape[0],g_1.shape[1],g_1.shape[2],g_1.shape[3],self.noise_base, device=g_1.device)
                #    g_1 = torch.cat([g_1, noise_base], dim=-1)
                if self.noise_base >= 1:
                    g_1_norm = torch.sum(g_1 ** 2, dim=(-2, -3), keepdim=True) ** 0.5
                    g_1 = g_1 / torch.maximum(g_1_norm, torch.ones_like(
                        g_1_norm) * 1e-4)  # + self.noise_level*noise/torch.sum( noise**2, dim = (-1,-2) , keepdim=True )
                else:
                    g_1_norm = torch.sum(g_1 ** 2, dim=(-2, -3, -4), keepdim=True) ** 0.5
                    g_1 = g_1 / torch.maximum(g_1_norm, torch.ones_like(
                        g_1_norm) * 1e-4)  # + self.noise_level*noise/torch.sum( noise**2, dim = (-1,-2) , keepdim=True )
            # Compute regularization level
            reg_Acc = (reg_level * self.reg_w) ** 0.5
            reg_target = ((reg_target_level * self.reg_w) ** 0.5).to(reg_Acc.device)
            # Compute residual
            g_flat_res = g.reshape(dxs.shape[0], -1)
            reg_g = self.reg_size * (reg_Acc[:, None] - reg_target[:, None])
            g_flat_res_reg = torch.cat((g_flat_res, reg_g), dim=-1)

            res_x = ((torch.mean((g_flat_res) ** 2, dim=(-1), keepdim=False)) ** 0.5)[:, None, None, None]
            res_el = ((torch.mean((g_flat_res_reg) ** 2, dim=(-1), keepdim=False)) ** 0.5)[:, None, None, None]
            # reg_res = torch.mean( (self.reg_size*torch.abs(reg_level - reg_target))**2 )**0.5
            # reg_res = torch.mean( self.reg_size*torch.abs(reg_level - self.reg_level)/g.shape[-1]/g.shape[-2] )**0.5

            res = torch.mean(res_el)  # + reg_res
            # if res < best_res:
            #    best_res = res
            #    best_dxs = dxs

            if iteration == 0:
                best_res_el = res_el
                best_dxs = dxs
                not_converged = torch.ones_like(res_el).bool()
            # update eps if residual is better
            res_mask = torch.logical_and(res_el < best_res_el, not_converged).int()
            best_res_el = res_mask * res_el + (1 - res_mask) * best_res_el
            # print(res_mask.shape, dxs.shape, best_dxs.shape)
            best_dxs = res_mask * dxs + (1 - res_mask) * best_dxs
            # eps0_ch, eps1_ch = res_mask*pred_eps_uncond + (1-res_mask)*eps0_ch, res_mask*pred_eps_cond + (1-res_mask)*eps1_ch

            res_max = torch.max(best_res_el)
            # print("res_x",  torch.max( res_x ), "reg", torch.max( reg_level), "reg_target", reg_target, "res", res_max )
            not_converged = torch.logical_and(res_el >= res_thres, not_converged)
            # torch._dynamo.graph_break()
            if res_max < res_thres:
                Converged = True
                break
            # v = beta*v + (1-beta)*g**2
            # m = beta_m*m + (1-beta_m)*g
            # g/(v**0.5+eps_delta)
            if self.noise_base >= 1:
                aa_dim = self.aa_dim
            else:
                aa_dim = 1
            dxs_Acc, g_Acc, reg_dxs_Acc, reg_g_Acc = AndersonAccR(dxs, g, reg_Acc, reg_target, pre_condition=None,
                                                                  m=aa_dim + 1)
            # print(Accout)
            #
            dxs = dxs_Acc - self.lr_chara * g_Acc
            reg_Acc = reg_dxs_Acc - self.lr_chara * reg_g_Acc
            reg_level = reg_Acc ** 2 / self.reg_w

            # reg_target_level = (1+self.reg_level)**( iteration//int(5/self.lr_chara) ) - 1
            # reg_level_mask = (reg_level >= reg_target_level).long()
            # reg_level = reg_level_mask*reg_level + (1-reg_level_mask)*reg_target_level
            # if iteration%int(5) == 0:
            #    dxs_Anderson = []
            #    g_Anderson = []
            iteration_counts = iteration_counts * (1 - not_converged.long()) + iteration * not_converged.long()

        self.ite_infos[0].append(best_res_el)
        self.ite_infos[1].append(iteration_counts[:, 0, 0, 0])
        self.ite_infos[2].append(reg_target_level)
        print("Characteristic iteration happens", iteration_counts[:, 0, 0, 0], "times")
        final_dxs = best_dxs * (1 - not_converged.long())
        dxs_cond_part = torch.cat([*([(h - 1) * final_dxs[:, None, ...]] * num_x_in_cond)], axis=1).view(
            (dxs.shape[0] * num_x_in_cond, *dxs.shape[1:]))
        dxs_add = torch.cat([dxs_cond_part, h * final_dxs], axis=0)
        # dxs_add = torch.cat([ *( [(h - 1) * final_dxs,]*num_x_in_cond ), h * final_dxs], axis=0)
        self.dxs_buffer = final_dxs
        self.abt_buffer = abt_current

        return evaluation(x_out_evaluation, x_in + dxs_add, sigma_in, **c)
    ##############################################################################
    out_cond = torch.zeros_like(x_in)
    out_count = torch.ones_like(x_in) * 1e-37

    out_uncond = torch.zeros_like(x_in)
    out_uncond_count = torch.ones_like(x_in) * 1e-37

    COND = 0
    UNCOND = 1

    to_run = []
    for x in cond:
        p = get_area_and_mult(x, x_in, timestep) #'cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'control', 'patches']
        if p is None:
            continue

        to_run += [(p, COND)]
    if uncond is not None:
        for x in uncond:
            p = get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue

            to_run += [(p, UNCOND)]

    while len(to_run) > 0:
        first = to_run[0] #[p,COND]
        first_shape = first[0][0].shape #p[0].shape, i.e., input_x.shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        free_memory = model_management.get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[:len(to_batch_temp)//i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            # print('condition or uncondition',o[1])
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        transformer_options = {}
        if 'transformer_options' in model_options:
            transformer_options = model_options['transformer_options'].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        transformer_options["cond_mark"] = compute_cond_mark(cond_or_uncond=cond_or_uncond, sigmas=timestep)
        transformer_options["cond_indices"], transformer_options["uncond_indices"] = compute_cond_indices(cond_or_uncond=cond_or_uncond, sigmas=timestep)

        c['transformer_options'] = transformer_options

        if control is not None:
            p = control
            while p is not None:
                p.transformer_options = transformer_options
                p = p.previous_controlnet
            control_cond = c.copy()  # get_control may change items in this dict, so we need to copy it
            c['control'] = control.get_control(input_x, timestep_, control_cond, len(cond_or_uncond))
            c['control_model'] = control
        # print('input_x',input_x.shape)
        # print('timestep',timestep_)
        # input size: (2*B,4,64,64) B: batch size
        #  timestep size: (2*B)
        # print('c',c.keys())
        # print('model',model)
        # print('uncond',uncond.shape)
        if 'model_function_wrapper' in model_options:
            output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
        else:
            #  timestep_: sigma_in
            # print('uncond',uncond)
            output =  Chara_iteration(self,model,None,input_x,timestep_,cond_scale,uncond[0]['cross_attn'],c).chunk(batch_chunks)
            # print('c',c['c_crossattn'].shape)
            # output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks) ###### apply model ###### output: len=2, 0 cond:torch.Size([B, 4, 64, 64]) 1 uncond: ([B, 4, 64, 64])
        del input_x
        # print('output',len(output))
        # print('output 1', output[1].shape)
        # print('cond_or_uncond: 0',cond_or_uncond[0])
        # print('cond_or_uncond: 1', cond_or_uncond[1])
        for o in range(batch_chunks):
            # print(f'{o} cond_or_uncond is {cond_or_uncond[o]}')
            if cond_or_uncond[o] == COND:
                out_cond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
            else:
                out_uncond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_uncond_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
        del mult

    out_cond /= out_count
    del out_count
    out_uncond /= out_uncond_count
    del out_uncond_count
    return out_cond, out_uncond
def sampling_function(self,model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
    print('*********** running sampling function *********** ')
    edit_strength = sum((item['strength'] if 'strength' in item else 1) for item in cond)
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    for fn in model_options.get("sampler_pre_cfg_function", []):
        # print('time step before',timestep)
        model, cond, uncond_, x, timestep, model_options = fn(model, cond, uncond_, x, timestep, model_options)
        # print('time step after', timestep)
    #this is where neural network evaluation happends at x
    cond_pred, uncond_pred = calc_cond_uncond_batch(self,model, cond, uncond_, x, timestep, model_options,cond_scale)
    # print('*' * 50)
    # all_attributes = dir(model)
    # print(all_attributes)
    # print(model.get_scalings)
    # callable_functions = [attr for attr in all_attributes if
    #                       callable(getattr(model, attr)) and not attr.startswith('__')]
    # print(callable_functions)
    # print('*' * 50)
    print( "Model type: ",model.model_type )
    # print("cond_pred", cond_pred.shape)
    # print("uncond_pred",cond_pred.shape)

    if "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    elif not math.isclose(edit_strength, 1.0):
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * edit_strength
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)
    print("**********Inject Sampling***********")
    return cfg_result

#forge_sample_str = inspect.getsource(forge_sampler.forge_sample)
#exec( forge_sample_str )


def forge_sample(self, denoiser_params, cond_scale, cond_composition):
    model = self.inner_model.inner_model.forge_objects.unet.model
    control = self.inner_model.inner_model.forge_objects.unet.controlnet_linked_list
    extra_concat_condition = self.inner_model.inner_model.forge_objects.unet.extra_concat_condition
    x = denoiser_params.x
    timestep = denoiser_params.sigma
    # print('a111 uncond',denoiser_params.text_uncond.shape)
    uncond = cond_from_a1111_to_patched_ldm(denoiser_params.text_uncond)
    # print('patched ldm uncond cross_attn', uncond[0]['cross_attn'].shape)
    # print('patched ldm uncond c_cross_attn', uncond[0]['model_conds']['c_crossattn'])
    # print('patched ldm uncond', (uncond))
    cond = cond_from_a1111_to_patched_ldm_weighted(denoiser_params.text_cond, cond_composition)
    # print('original cond', (denoiser_params.text_cond).shape)
    # print('patched ldm cond', (cond))
    model_options = self.inner_model.inner_model.forge_objects.unet.model_options
    seed = self.p.seeds[0]
    if extra_concat_condition is not None:
        image_cond_in = extra_concat_condition
    else:
        image_cond_in = denoiser_params.image_cond

    if isinstance(image_cond_in, torch.Tensor):
        if image_cond_in.shape[0] == x.shape[0] \
                and image_cond_in.shape[2] == x.shape[2] \
                and image_cond_in.shape[3] == x.shape[3]:
            for i in range(len(uncond)):
                uncond[i]['model_conds']['c_concat'] = CONDRegular(image_cond_in)
            for i in range(len(cond)):
                cond[i]['model_conds']['c_concat'] = CONDRegular(image_cond_in)

    if control is not None:
        for h in cond + uncond:
            h['control'] = control

    for modifier in model_options.get('conditioning_modifiers', []):
        model, x, timestep, uncond, cond, cond_scale, model_options, seed = modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed)

    denoised = sampling_function(self,model, x, timestep, uncond, cond, cond_scale, model_options, seed)
    return denoised


def CHGdenoiserConstruct():
    CHGDenoiserStr = '''
class CHGDenoiser(CFGDenoiser):
    def __init__(self, sampler):
        super().__init__(sampler)
    def forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
        #self.inner_model: CompVisDenoiser
        if state.interrupted or state.skipped:
            raise sd_samplers_common.InterruptedException

        original_x_device = x.device
        original_x_dtype = x.dtype
        acd = self.inner_model.inner_model.alphas_cumprod

        if self.classic_ddim_eps_estimation:
            acd = self.inner_model.inner_model.alphas_cumprod
            fake_sigmas = ((1 - acd) / acd) ** 0.5
            real_sigma = fake_sigmas[sigma.round().long().clip(0, int(fake_sigmas.shape[0]))]
            real_sigma_data = 1.0
            x = x * (((real_sigma ** 2.0 + real_sigma_data ** 2.0) ** 0.5)[:, None, None, None])
            sigma = real_sigma

        if sd_samplers_common.apply_refiner(self, x):
            cond = self.sampler.sampler_extra_args['cond']
            uncond = self.sampler.sampler_extra_args['uncond']

        cond_composition, cond = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)

        if self.mask is not None:
            noisy_initial_latent = self.init_latent + sigma[:, None, None, None] * torch.randn_like(self.init_latent).to(self.init_latent)
            x = x * self.nmask + noisy_initial_latent * self.mask

        denoiser_params = CFGDenoiserParams(x, image_cond, sigma, state.sampling_step, state.sampling_steps, cond, uncond, self)
        cfg_denoiser_callback(denoiser_params)

        denoised = forge_inject.forge_sample(self,denoiser_params=denoiser_params,
                                              cond_scale=cond_scale, cond_composition=cond_composition)

        if self.mask is not None:
            denoised = denoised * self.nmask + self.init_latent * self.mask

        preview = self.sampler.last_latent = denoised
        sd_samplers_common.store_latent(preview)

        after_cfg_callback_params = AfterCFGCallbackParams(denoised, state.sampling_step, state.sampling_steps)
        cfg_after_cfg_callback(after_cfg_callback_params)
        denoised = after_cfg_callback_params.x

        self.step += 1

        if self.classic_ddim_eps_estimation:
            eps = (x - denoised) / sigma[:, None, None, None]
            return eps
        print("*****CHG inject success*****")

        return denoised.to(device=original_x_device, dtype=original_x_dtype)
'''
    #CHGDenoiserStr += inspect.getsource(CFGDenoiser.forward)
    print(CHGDenoiserStr)
    return CHGDenoiserStr