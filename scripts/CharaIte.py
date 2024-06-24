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
import k_diffusion.utils as utils
from k_diffusion.external import CompVisVDenoiser, CompVisDenoiser
from modules.sd_samplers_timesteps import CompVisTimestepsDenoiser, CompVisTimestepsVDenoiser
from modules.sd_samplers_cfg_denoiser import CFGDenoiser, catenate_conds, subscript_cond, pad_cond
from modules import script_callbacks

try:
    from modules_forge import forge_sampler
    isForge = True
except Exception:
    isForge = False

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


def Chara_iteration(self, *args, **kwargs):
    if not isForge:
        dxs, x_in, sigma_in, tensor, uncond, cond_scale, image_cond_in, is_edit_model, skip_uncond, make_condition_dict, batch_cond_uncond, batch_size = args 
        cond_in=kwargs["cond_in"]
        x_out = kwargs["x_out"]
        # function being evaluated must have x_in and cond_in as first and second input
        def x_out_evaluation(x_in, cond_in, sigma_in, image_cond_in):
            return self.inner_model(x_in, sigma_in, cond=make_condition_dict(cond_in, image_cond_in))
        def eps_evaluation(x_in, cond_in, t_in, image_cond_in):
            return self.inner_model.get_eps(x_in, t_in, cond=make_condition_dict(cond_in, image_cond_in))
        def v_evaluation(x_in, cond_in, t_in, image_cond_in):
            return self.inner_model.get_v(x_in, t_in, cond=make_condition_dict(cond_in, image_cond_in))
        def eps_legacy_evaluation(x_in, cond_in, t_in, image_cond_in):
            return self.inner_model(x_in, t_in, cond=make_condition_dict(cond_in, image_cond_in))
        if tensor.shape[1] == uncond.shape[1] or skip_uncond:
            if batch_cond_uncond:
                def evaluation(func, x_in, conds, *args, **kwargs):
                    tensor, uncond, cond_in = conds
                    return func(x_in, cond_in, *args, **kwargs)
            else:
                def evaluation(func, x_in, conds, *args, **kwargs):
                    x_out = torch.zeros_like(x_in)
                    tensor, uncond, cond_in = conds
                    for batch_offset in range(0, x_out.shape[0], batch_size):
                        a = batch_offset
                        b = a + batch_size
                        x_out[a:b] = func(x_in[a:b],subscript_cond(cond_in, a, b), *[arg[a:b] for arg in args], **kwargs)
                    return x_out
        else:
            def evaluation(func, x_in, conds, *args, **kwargs):
                x_out = torch.zeros_like(x_in)
                tensor, uncond, cond_in = conds
                batch_Size = batch_size*2 if batch_cond_uncond else batch_size
                for batch_offset in range(0, tensor.shape[0], batch_Size):
                    a = batch_offset
                    b = min(a + batch_Size, tensor.shape[0])

                    if not is_edit_model:
                        c_crossattn = subscript_cond(tensor, a, b)
                    else:
                        c_crossattn = torch.cat([tensor[a:b]], uncond)

                    x_out[a:b] = func(x_in[a:b], c_crossattn, *[arg[a:b] for arg in args], **kwargs)

                if not skip_uncond:
                    x_out[-uncond.shape[0]:] = func(x_in[-uncond.shape[0]:], uncond, *[arg[-uncond.shape[0]:] for arg in args], **kwargs)

                return x_out
        if is_edit_model or skip_uncond:
            return evaluation(x_out_evaluation, x_in, (tensor, uncond, cond_in), sigma_in, image_cond_in)
        else:
            evaluations = [eps_evaluation, v_evaluation, eps_legacy_evaluation, evaluation]
            ite_paras = [dxs, x_in, sigma_in, tensor, uncond, cond_scale, image_cond_in, is_edit_model, skip_uncond, make_condition_dict, batch_cond_uncond, batch_size, cond_in, x_out]
            dxs_add = chara_ite_inner_loop(self, evaluations, ite_paras)
            return evaluation(x_out_evaluation, x_in + dxs_add, (tensor, uncond, cond_in), sigma_in, image_cond_in)
    else:
        model,dxs,x_in, sigma_in,cond_scale,uncond, c = args
        def evaluation(func, x_in, t_in, c_concat=None, c_crossattn=None, control=None, transformer_options={}, *args, **kwargs):
            # tensor, uncond, cond_in = conds
            # print('x_in eval',x_in.shape)
            return func(x_in, t_in, c_concat, c_crossattn, control, transformer_options, *args, **kwargs)

        def eps_evaluation(x_in, t_in, c_concat=None, c_crossattn=None, control=None, transformer_options={}):
            # print('c concat',c_concat.shape)
            # print('c_crossattn 0',(c_crossattn)[0][:10])
            # print('c_crossattn 1', (c_crossattn)[1][:10])
            #print('model eps evaluation')
            x_out = model.apply_model(x_in,t_in, c_concat, c_crossattn, control, transformer_options)
            t_in_expand = t_in.view(t_in.shape[:1] + (1,) * (x_in.ndim - 1))
            eps_out = (x_in - x_out)/t_in_expand
            return eps_out

        def v_evaluation(x_in, t_in, c_concat=None, c_crossattn=None, control=None, transformer_options={}):
            #print('model v evaluation')
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
        evaluations = [eps_evaluation, v_evaluation, None, evaluation]
        ite_paras = [model,dxs,x_in, sigma_in,cond_scale,uncond, c]
        dxs_add = chara_ite_inner_loop(self, evaluations, ite_paras)
        return evaluation(x_out_evaluation, x_in + dxs_add, sigma_in, **c)

def chara_ite_inner_loop(self, evaluations, ite_paras):
    eps_evaluation, v_evaluation, eps_legacy_evaluation, evaluation = evaluations
    if isForge:
        model,dxs,x_in, sigma_in,cond_scale,uncond, c = ite_paras
    else:
        dxs, x_in, sigma_in, tensor, uncond, cond_scale, image_cond_in, is_edit_model, skip_uncond, make_condition_dict, batch_cond_uncond, batch_size, cond_in, x_out = ite_paras
    if dxs is None:
        dxs = torch.zeros_like(x_in[-uncond.shape[0]:])
    if self.radio_controlnet == "More Prompt":
        control_net_weights = []
        for script in self.process_p.scripts.scripts:
            if script.title() == "ControlNet":
                try:
                    for param in script.latest_network.control_params:
                        control_net_weights.append(param.weight)
                        param.weight = 0.
                except:
                    pass

    res_thres = self.res_thres
    
    num_x_in_cond = len(x_in[:-uncond.shape[0]])//len(dxs)
    
    h = cond_scale*num_x_in_cond

    if isinstance(self.inner_model, CompVisDenoiser):
        t_in = self.inner_model.sigma_to_t(sigma_in)
        abt = self.inner_model.inner_model.alphas_cumprod.to(t_in.device)[t_in.long()]
        c_out, c_in = [utils.append_dims(x, x_in.ndim) for x in self.inner_model.get_scalings(sigma_in)]
    elif isinstance(self.inner_model, CompVisVDenoiser):
        t_in = self.inner_model.sigma_to_t(sigma_in)
        abt = self.inner_model.inner_model.alphas_cumprod.to(t_in.device)[t_in.long()]
        c_skip, c_out, c_in = [utils.append_dims(x, x_in.ndim) for x in self.inner_model.get_scalings(sigma_in)]
    elif isinstance(self.inner_model, CompVisTimestepsDenoiser) or isinstance(self.inner_model,
                                                                                CompVisTimestepsVDenoiser):
        t_in = sigma_in
        abt = self.alphas[t_in.long()]
    else:
        raise NotImplementedError()


    scale = ((1 - abt) ** 0.5)[-uncond.shape[0]:, None, None, None].to(x_in.device)
    abt_current = abt[-uncond.shape[0]:, None, None, None].to(x_in.device)
    abt_smallest = self.inner_model.inner_model.alphas_cumprod[-1].to(x_in.device)
    # x_in_cond = x_in[:-uncond.shape[0]]
    # x_in_uncond = x_in[-uncond.shape[0]:]
    # print("alphas_cumprod",-torch.log(self.inner_model.inner_model.alphas_cumprod))
    # print("betas",torch.sum(self.inner_model.inner_model.betas))

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
            if torch.sum( torch.isnan(gamma) ) > 0:
                gamma = solve_least_squares(gA/gA_norm, gB)
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
            A = g_1.reshape(g_1.shape[0], g_1.shape[1]* g_1.shape[2] * g_1.shape[3], g_1.shape[4])
            B = dx.reshape(dx.shape[0], -1, 1)
            regl = reg[:, None].reshape(dx.shape[0], 1, 1)
            dx_proj = proj_least_squares(A, B, regl)

            return dx_proj.reshape(*dx.shape)
    g_1 = None

    reg_level = torch.zeros(dxs.shape[0], device=dxs.device) + max(5,self.reg_ini)
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
        dxs = dxs * ((abt_prev - abt_current * abt_prev) / (abt_current - abt_current * abt_prev))
        # print(abt_prev.shape, abt_current.shape, self.dxs_buffer.shape)
        dxs = self.chara_decay * dxs
    iteration_counts = 0
    for iteration in range(n_iterations):
        # important to keep iteration content consistent
        # Supoort AND prompt combination by using multiple dxs for condition part
        def compute_correction_direction(dxs):
            dxs_cond_part = torch.cat( [*( [(h - 1) * dxs[:,None,...]]*num_x_in_cond )], axis=1 ).view( (dxs.shape[0]*num_x_in_cond, *dxs.shape[1:]) )
            dxs_add = torch.cat([ dxs_cond_part, h * dxs], axis=0)
            if isinstance(self.inner_model, CompVisDenoiser):
                if isForge:
                    eps_out = evaluation(eps_evaluation, x_in + dxs_add, sigma_in,**c)
                    pred_eps_uncond = eps_out[:-uncond.shape[0]] # forge: c_crossatten[0]: uncondition
                    eps_cond_batch = eps_out[-uncond.shape[0]:] # forge: c_crossatten[1]: condition
                else:
                    eps_out = evaluation(eps_evaluation, x_in * c_in + dxs_add * c_in, (tensor, uncond, cond_in), t_in, image_cond_in)
                    pred_eps_uncond = eps_out[-uncond.shape[0]:]
                    eps_cond_batch = eps_out[:-uncond.shape[0]]
                eps_cond_batch_target_shape = ( len(eps_cond_batch)//num_x_in_cond, num_x_in_cond, *(eps_cond_batch.shape[1:]) )
                pred_eps_cond = torch.mean( eps_cond_batch.view(eps_cond_batch_target_shape), dim=1, keepdim=False )
                ggg = (pred_eps_uncond - pred_eps_cond) * scale / c_in[-uncond.shape[0]:]
            elif isinstance(self.inner_model, CompVisVDenoiser):
                if isForge:
                    v_out = evaluation(v_evaluation, x_in+dxs_add,sigma_in, **c)
                    eps_out = -c_out*x_in + c_skip**0.5*v_out
                    pred_eps_uncond = eps_out[:-uncond.shape[0]] # forge: c_crossatten[0]: uncondition
                    eps_cond_batch = eps_out[-uncond.shape[0]:] # forge: c_crossatten[1]: condition
                else:
                    v_out = evaluation(v_evaluation, x_in * c_in + dxs_add * c_in, (tensor, uncond, cond_in), t_in, image_cond_in)
                    eps_out = -c_out*x_in + c_skip**0.5*v_out
                    pred_eps_uncond = eps_out[-uncond.shape[0]:]
                    eps_cond_batch = eps_out[:-uncond.shape[0]]
                eps_cond_batch_target_shape = ( len(eps_cond_batch)//num_x_in_cond, num_x_in_cond, *(eps_cond_batch.shape[1:]) )
                pred_eps_cond = torch.mean( eps_cond_batch.view(eps_cond_batch_target_shape), dim=1, keepdim=False )
                ggg = (pred_eps_uncond - pred_eps_cond) * scale / c_in[-uncond.shape[0]:]
            elif isinstance(self.inner_model, CompVisTimestepsDenoiser) or isinstance(self.inner_model,
                                                                                    CompVisTimestepsVDenoiser):
                #eps_out = self.inner_model(x_in + dxs_add, t_in, cond=cond)
                eps_out = evaluation(eps_legacy_evaluation, x_in + dxs_add, (tensor, uncond, cond_in), t_in, image_cond_in)
                pred_eps_uncond = eps_out[-uncond.shape[0]:]
                eps_cond_batch = eps_out[:-uncond.shape[0]]
                eps_cond_batch_target_shape = ( len(eps_cond_batch)//num_x_in_cond, num_x_in_cond, *(eps_cond_batch.shape[1:]) )
                pred_eps_cond = torch.mean( eps_cond_batch.view(eps_cond_batch_target_shape), dim=1, keepdim=False )
                ggg = (pred_eps_uncond - pred_eps_cond) * scale
            else:
                raise NotImplementedError()
            return ggg

        ggg = compute_correction_direction(dxs)
        # print("print(reg_level.shape)", reg_level.shape)
        g = dxs - downsample_reg_g(ggg, g_1, reg_level)
        if g_1 is None:
            g_basis = -compute_correction_direction(dxs*0)
            g_1 = split_basis(g_basis, max( self.noise_base,1 ) )
            # if self.Projg:
            #        g_1 = split_basis( g, self.noise_base)
            # else:
            #        g_1 = split_basis( ggg, self.noise_base)
            # if self.CFGdecayS and self.dxs_buffer is not None:
            #     g_1 = torch.cat( [g_1, self.dxs_buffer[:,:,:,:,None]], dim=-1 )
            # if self.noise_base > 0:
            #    noise_base = torch.randn(g_1.shape[0],g_1.shape[1],g_1.shape[2],g_1.shape[3],self.noise_base, device=g_1.device)
            #    g_1 = torch.cat([g_1, noise_base], dim=-1)
            if self.noise_base >=1:
                g_1_norm = torch.sum(g_1 ** 2, dim=(-2, -3), keepdim=True) ** 0.5
                g_1 = g_1 / torch.maximum(g_1_norm, torch.ones_like(
                    g_1_norm) * 1e-4)  # + self.noise_level*noise/torch.sum( noise**2, dim = (-1,-2) , keepdim=True )
            else:
                g_1_norm = torch.sum(g_1 ** 2, dim=(-2, -3, -4), keepdim=True) ** 0.5
                g_1 = g_1 / torch.maximum(g_1_norm, torch.ones_like(
                    g_1_norm) * 1e-4)  # + self.noise_level*noise/torch.sum( noise**2, dim = (-1,-2) , keepdim=True )
        # Compute regularization level
        reg_Acc = (reg_level * self.reg_w) ** 0.5
        reg_target = (reg_target_level * self.reg_w) ** 0.5
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
        # print("not_converged", not_converged.shape)
        # torch._dynamo.graph_break()
        if res_max < res_thres:
            Converged = True
            break
        # v = beta*v + (1-beta)*g**2
        # m = beta_m*m + (1-beta_m)*g
        # g/(v**0.5+eps_delta)
        if self.noise_base >=1:
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
    # print(iteration_counts[:,0,0,0].shape)
    self.ite_infos[1].append(iteration_counts[:, 0, 0, 0])
    self.ite_infos[2].append(reg_target_level)
    print("Characteristic iteration happens", iteration_counts[:, 0, 0, 0] , "times")
    final_dxs = best_dxs * (1 - not_converged.long())
    dxs_cond_part = torch.cat( [*( [(h - 1) * final_dxs[:,None,...]]*num_x_in_cond )], axis=1 ).view( (dxs.shape[0]*num_x_in_cond, *dxs.shape[1:]) )
    dxs_add = torch.cat([ dxs_cond_part, h * final_dxs], axis=0)
    #dxs_add = torch.cat([ *( [(h - 1) * final_dxs,]*num_x_in_cond ), h * final_dxs], axis=0)
    self.dxs_buffer = final_dxs
    self.abt_buffer = abt_current

    if self.radio_controlnet == "More Prompt":
        controlnet_count = 0
        for script in self.process_p.scripts.scripts:
            if script.title() == "ControlNet":
                try:
                    for param in script.latest_network.control_params:
                        param.weight = control_net_weights[controlnet_count]
                        controlnet_count += 1
                except:
                    pass          
    return dxs_add              