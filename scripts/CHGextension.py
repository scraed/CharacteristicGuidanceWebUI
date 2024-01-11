import modules.scripts as scripts
import gradio as gr

import io
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
from modules import prompt_parser, devices, sd_samplers_common

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
######## Infotext processing ##########
quote_swap = str.maketrans('\'"', '"\'')


def pares_infotext(infotext, params):
    # parse infotext decode json string
    try:
        params['CHG'] = json.loads(params['CHG'].translate(quote_swap))
    except Exception:
        pass


script_callbacks.on_infotext_pasted(pares_infotext)
#######################################

class CHGDenoiser(CFGDenoiser):
    def __init__(self, sampler):
        super().__init__(sampler)

    def forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
        if state.interrupted or state.skipped:
            raise sd_samplers_common.InterruptedException

        if sd_samplers_common.apply_refiner(self):
            cond = self.sampler.sampler_extra_args['cond']
            uncond = self.sampler.sampler_extra_args['uncond']

        # at self.image_cfg_scale == 1.0 produced results for edit model are the same as with normal sampling,
        # so is_edit_model is set to False to support AND composition.
        is_edit_model = shared.sd_model.cond_stage_key == "edit" and self.image_cfg_scale is not None and self.image_cfg_scale != 1.0

        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)

        assert not is_edit_model or all(len(conds) == 1 for conds in
                                        conds_list), "AND is not supported for InstructPix2Pix checkpoint (unless using Image CFG scale = 1.0)"

        if self.mask_before_denoising and self.mask is not None:
            x = self.init_latent * self.mask + self.nmask * x

        batch_size = len(conds_list)
        repeats = [len(conds_list[i]) for i in range(batch_size)]

        if shared.sd_model.model.conditioning_key == "crossattn-adm":
            image_uncond = torch.zeros_like(image_cond)
            make_condition_dict = lambda c_crossattn, c_adm: {"c_crossattn": [c_crossattn], "c_adm": c_adm}
        else:
            image_uncond = image_cond
            if isinstance(uncond, dict):
                make_condition_dict = lambda c_crossattn, c_concat: {**c_crossattn, "c_concat": [c_concat]}
            else:
                make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": [c_crossattn],
                                                                     "c_concat": [c_concat]}

        if not is_edit_model:
            x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x])
            sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma])
            image_cond_in = torch.cat(
                [torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_uncond])
        else:
            x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x] + [x])
            sigma_in = torch.cat(
                [torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma] + [sigma])
            image_cond_in = torch.cat(
                [torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_uncond] + [
                    torch.zeros_like(self.init_latent)])

        denoiser_params = CFGDenoiserParams(x_in, image_cond_in, sigma_in, state.sampling_step, state.sampling_steps,
                                            tensor, uncond)
        cfg_denoiser_callback(denoiser_params)
        x_in = denoiser_params.x
        image_cond_in = denoiser_params.image_cond
        sigma_in = denoiser_params.sigma
        tensor = denoiser_params.text_cond
        uncond = denoiser_params.text_uncond
        skip_uncond = False

        # alternating uncond allows for higher thresholds without the quality loss normally expected from raising it
        if self.step % 2 and s_min_uncond > 0 and sigma[0] < s_min_uncond and not is_edit_model:
            skip_uncond = True
            x_in = x_in[:-batch_size]
            sigma_in = sigma_in[:-batch_size]
            print("***********************************skip uncond***************")

        self.padded_cond_uncond = False
        if shared.opts.pad_cond_uncond and tensor.shape[1] != uncond.shape[1]:
            empty = shared.sd_model.cond_stage_model_empty_prompt
            num_repeats = (tensor.shape[1] - uncond.shape[1]) // empty.shape[1]

            if num_repeats < 0:
                tensor = pad_cond(tensor, -num_repeats, empty)
                self.padded_cond_uncond = True
            elif num_repeats > 0:
                uncond = pad_cond(uncond, num_repeats, empty)
                self.padded_cond_uncond = True

        if tensor.shape[1] == uncond.shape[1] or skip_uncond:
            if is_edit_model:
                cond_in = catenate_conds([tensor, uncond, uncond])
            elif skip_uncond:
                cond_in = tensor
            else:
                cond_in = catenate_conds([tensor, uncond])

            if shared.opts.batch_cond_uncond:
                x_out = self.Chara_iteration(None, x_in, sigma_in, uncond, cond_scale, conds_list,
                                             cond=make_condition_dict(cond_in, image_cond_in))
            else:
                x_out = torch.zeros_like(x_in)
                for batch_offset in range(0, x_out.shape[0], batch_size):
                    a = batch_offset
                    b = a + batch_size
                    x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b],
                                                  cond=make_condition_dict(subscript_cond(cond_in, a, b),
                                                                           image_cond_in[a:b]))
        else:
            x_out = torch.zeros_like(x_in)
            batch_size = batch_size * 2 if shared.opts.batch_cond_uncond else batch_size
            for batch_offset in range(0, tensor.shape[0], batch_size):
                a = batch_offset
                b = min(a + batch_size, tensor.shape[0])

                if not is_edit_model:
                    c_crossattn = subscript_cond(tensor, a, b)
                else:
                    c_crossattn = torch.cat([tensor[a:b]], uncond)

                x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b],
                                              cond=make_condition_dict(c_crossattn, image_cond_in[a:b]))

            if not skip_uncond:
                x_out[-uncond.shape[0]:] = self.inner_model(x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:],
                                                            cond=make_condition_dict(uncond,
                                                                                     image_cond_in[-uncond.shape[0]:]))

        denoised_image_indexes = [x[0][0] for x in conds_list]
        if skip_uncond:
            fake_uncond = torch.cat([x_out[i:i + 1] for i in denoised_image_indexes])
            x_out = torch.cat([x_out,
                               fake_uncond])  # we skipped uncond denoising, so we put cond-denoised image to where the uncond-denoised image should be

        denoised_params = CFGDenoisedParams(x_out, state.sampling_step, state.sampling_steps, self.inner_model)
        cfg_denoised_callback(denoised_params)

        devices.test_for_nans(x_out, "unet")

        if is_edit_model:
            denoised = self.combine_denoised_for_edit_model(x_out, cond_scale)
        elif skip_uncond:
            denoised = self.combine_denoised(x_out, conds_list, uncond, 1.0)
        else:
            denoised = self.combine_denoised(x_out, conds_list, uncond, cond_scale)

        if not self.mask_before_denoising and self.mask is not None:
            denoised = self.init_latent * self.mask + self.nmask * denoised

        self.sampler.last_latent = self.get_pred_x0(torch.cat([x_in[i:i + 1] for i in denoised_image_indexes]),
                                                    torch.cat([x_out[i:i + 1] for i in denoised_image_indexes]), sigma)

        if opts.live_preview_content == "Prompt":
            preview = self.sampler.last_latent
        elif opts.live_preview_content == "Negative prompt":
            preview = self.get_pred_x0(x_in[-uncond.shape[0]:], x_out[-uncond.shape[0]:], sigma)
        else:
            preview = self.get_pred_x0(torch.cat([x_in[i:i + 1] for i in denoised_image_indexes]),
                                       torch.cat([denoised[i:i + 1] for i in denoised_image_indexes]), sigma)

        sd_samplers_common.store_latent(preview)

        after_cfg_callback_params = AfterCFGCallbackParams(denoised, state.sampling_step, state.sampling_steps)
        cfg_after_cfg_callback(after_cfg_callback_params)
        denoised = after_cfg_callback_params.x

        self.step += 1
        return denoised

    def Chara_iteration(self, dxs, x_in, sigma_in, uncond, cond_scale, conds_list, cond=None):
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
        h = cond_scale

        # print("sigma_in", sigma_in)
        if isinstance(self.inner_model, CompVisDenoiser) or isinstance(self.inner_model, CompVisVDenoiser):
            t_in = self.inner_model.sigma_to_t(sigma_in)
            abt = self.inner_model.inner_model.alphas_cumprod[t_in.long()]
            c_out, c_in = [utils.append_dims(x, x_in.ndim) for x in self.inner_model.get_scalings(sigma_in)]
        elif isinstance(self.inner_model, CompVisTimestepsDenoiser) or isinstance(self.inner_model,
                                                                                  CompVisTimestepsVDenoiser):
            t_in = sigma_in
            abt = self.alphas[t_in.long()]
        else:
            raise NotImplementedError()

        scale = ((1 - abt) ** 0.5)[:-uncond.shape[0], None, None, None]
        abt_current = abt[:-uncond.shape[0], None, None, None]
        abt_smallest = self.inner_model.inner_model.alphas_cumprod[-1]
        # x_in_cond = x_in[:-uncond.shape[0]]
        # x_in_uncond = x_in[-uncond.shape[0]:]
        # print("alphas_cumprod",-torch.log(self.inner_model.inner_model.alphas_cumprod))
        # print("betas",torch.sum(self.inner_model.inner_model.betas))

        dxs_Anderson = []
        g_Anderson = []

        def solve_least_squares(A, B):
            # print(A.shape)
            # print(B.shape)
            # Compute C = A^T A
            # min_eigenvalues = torch.min( torch.linalg.eigvalsh(C), dim=-1 )
            # eps_e = torch.maximum( min_eigenvalues, min_eigenvalues.new_ones(min_eigenvalues.shape)*1e-3 )[...,]
            C = torch.matmul(A.transpose(-2, -1), A)  # + eps_e*torch.eye(A.shape[-1], device=A.device)
            # Compute the eigenvalues and eigenvectors of C
            eigenvalues, eigenvectors = torch.linalg.eigh(C)

            # eigenvalues = torch.maximum( eigenvalues,eigenvalues*0+1e-3  )

            # Diagonal matrix with non-zero eigenvalues in the diagonal
            D_inv = torch.diag_embed(1.0 / torch.maximum(eigenvalues, torch.ones_like(eigenvalues) * 1e-4))

            # Compute the pseudo-inverse of C
            C_inv = torch.matmul(torch.matmul(eigenvectors, D_inv), eigenvectors.transpose(-2, -1))

            # Compute X = C_inv A^T B
            X = torch.matmul(torch.matmul(C_inv, A.transpose(-2, -1)), B)
            return X

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

        g_1 = None

        def downsample_reg_g(dx, g_1, reg):
            # DDec_dx = DDec(dx)
            # down_DDec_dx = downsample(DDec_dx, factor=factor)
            # DEnc_dx = DEnc(down_DDec_dx)
            # return DEnc_dx

            if g_1 is None:
                return dx
            else:
                # return g_1*torch.sum(g_1*dx, dim = (-1,-2), keepdim=True )/torch.sum( g_1**2, dim = (-1,-2) , keepdim=True )
                A = g_1.reshape(g_1.shape[0] * g_1.shape[1], g_1.shape[2] * g_1.shape[3], g_1.shape[4])
                B = dx.reshape(dx.shape[0] * dx.shape[1], -1, 1)
                regl = reg[:, None].expand(-1, dx.shape[1]).reshape(dx.shape[0] * dx.shape[1], 1, 1)
                dx_proj = proj_least_squares(A, B, regl)

                return dx_proj.reshape(*dx.shape)

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
            dxs_add = torch.cat([(h - 1) * dxs, h * dxs], axis=0)
            if isinstance(self.inner_model, CompVisDenoiser) or isinstance(self.inner_model, CompVisVDenoiser):
                eps_out = self.inner_model.get_eps(x_in * c_in + dxs_add * c_in, t_in, cond=cond)
                pred_eps_cond, pred_eps_uncond = eps_out.chunk(2)
                ggg = (pred_eps_uncond - pred_eps_cond) * scale / c_in[-uncond.shape[0]:]
            elif isinstance(self.inner_model, CompVisTimestepsDenoiser) or isinstance(self.inner_model,
                                                                                      CompVisTimestepsVDenoiser):
                eps_out = self.inner_model(x_in + dxs_add, t_in, cond=cond)
                pred_eps_cond, pred_eps_uncond = eps_out.chunk(2)
                ggg = (pred_eps_uncond - pred_eps_cond) * scale
            else:
                raise NotImplementedError()

            # print("print(reg_level.shape)", reg_level.shape)
            g = dxs - downsample_reg_g(ggg, g_1, reg_level)
            if g_1 is None:
                g_1 = split_basis(g, self.noise_base)
                # if self.Projg:
                #        g_1 = split_basis( g, self.noise_base)
                # else:
                #        g_1 = split_basis( ggg, self.noise_base)
                # if self.CFGdecayS and self.dxs_buffer is not None:
                #     g_1 = torch.cat( [g_1, self.dxs_buffer[:,:,:,:,None]], dim=-1 )
                # if self.noise_base > 0:
                #    noise_base = torch.randn(g_1.shape[0],g_1.shape[1],g_1.shape[2],g_1.shape[3],self.noise_base, device=g_1.device)
                #    g_1 = torch.cat([g_1, noise_base], dim=-1)
                g_1_norm = torch.sum(g_1 ** 2, dim=(-2, -3), keepdim=True) ** 0.5
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

            dxs_Acc, g_Acc, reg_dxs_Acc, reg_g_Acc = AndersonAccR(dxs, g, reg_Acc, reg_target, pre_condition=None,
                                                                  m=self.aa_dim + 1)
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
        final_dxs = best_dxs * (1 - not_converged.long())
        dxs_add = torch.cat([(h - 1) * final_dxs, h * final_dxs], axis=0)
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

        return self.inner_model(x_in + dxs_add, sigma_in, cond=cond)


class ExtensionTemplateScript(scripts.Script):
    # Extension title in menu UI
    def title(self):
        return "Characteristic Guidance"

    # Decide to show menu in txt2img or img2img
    # - in "txt2img" -> is_img2img is `False`
    # - in "img2img" -> is_img2img is `True`
    #
    # below code always show extension menu
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def update_plot(self):
        from modules.sd_samplers_cfg_denoiser import CFGDenoiser
        try:
            res, ite_num, reg = CFGDenoiser.ite_infos
            res = np.array([r[:, 0, 0, 0].cpu().numpy() for r in res]).T
            ite_num = np.array([r.cpu().numpy() for r in ite_num]).T
            reg = np.array([r.cpu().numpy() for r in reg]).T
            if len(res) == 0:
                raise Exception('res has not been written yet')
        except Exception as e:
            res, ite_num, reg = [np.linspace(1, 0., 50)], [np.ones(50) * 10], [np.linspace(1, 0., 50)]
            print("The following exception occured when reading iteration info, demo plot is returned")
            print(e)


        try:
            res_thres = CFGDenoiser.res_thres
        except:
            res_thres = 0.1
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Converged'),
                           Patch(facecolor='yellow', label='Barely Converged'),
                           Patch(facecolor='red', label='Not Converged')]
        # Create bar plot
        fig, axs = plt.subplots(len(res), 1, figsize=(10, 4 * len(res)))
        if len(res) > 1:
            # Example plotting code
            for i in range(len(res)):
                # Categorize each result and assign colors
                colors = ['green' if r < res_thres else 'yellow' if r < 10 * res_thres else 'red' for r in res[i]]
                axs[i].bar(range(len(ite_num[i])), ite_num[i], color=colors)
                # Create legend
                axs[i].legend(handles=legend_elements, loc='upper right')

                # Add labels and title
                axs[i].set_xlabel('Diffusion Step')
                axs[i].set_ylabel('Num. Characteristic Iteration')
                ax2 = axs[i].twinx()
                ax2.plot(range(len(ite_num[i])), reg[i], linewidth=4, color='C1', label='Regularization Level')
                ax2.set_ylabel('Regularization Level')
                ax2.set_ylim(bottom=0.)
                ax2.legend(loc='upper left')
            # axs[i].set_title('Convergence Status of Iterations for Each Step')
        elif len(res) == 1:
            colors = ['green' if r < res_thres else 'yellow' if r < 10 * res_thres else 'red' for r in res[0]]
            axs.bar(range(len(ite_num[0])), ite_num[0], color=colors)
            # Create legend
            axs.legend(handles=legend_elements, loc='upper right')

            # Add labels and title
            axs.set_xlabel('Diffusion Step')
            axs.set_ylabel('Num. Characteristic Iteration')
            ax2 = axs.twinx()
            ax2.plot(range(len(ite_num[0])), reg[0], linewidth=4, color='C1', label='Regularization Level')
            ax2.set_ylabel('Regularization Level')
            ax2.set_ylim(bottom=0.)
            ax2.legend(loc='upper left')
        else:
            pass
            # axs.set_title('Convergence Status of Iterations for Each Step')
        # Convert the Matplotlib plot to a PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)

        plt.close()  # Close the plot
        return img

    # Setup menu ui detail
    def ui(self, is_img2img):
        with gr.Accordion('Characteristic Guidance', open=False):
            reg_ini = gr.Slider(
                minimum=0.0,
                maximum=10.,
                step=0.1,
                value=1.,
                label="Regularization Strength ( → Easier Convergence, Closer to CFG)",
            )
            reg_range = gr.Slider(
                minimum=0.01,
                maximum=10.,
                step=0.01,
                value=1.,
                label="Regularization Range Over Time ( ← Harder Convergence, More Correction)",
            )
            ite = gr.Slider(
                minimum=1,
                maximum=50,
                step=1,
                value=30,
                label="Max Num. Characteristic Iteration ( → Slow but Better Convergence)",
            )
            noise_base = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=1,
                label="Num. Basis for Correction ( ← Less Correction, Better Convergence)",
            )
            chara_decay = gr.Slider(
                minimum=0.,
                maximum=1.,
                step=0.01,
                value=0.,
                label="Reuse Correction of Previous Iteration ( → Suppress Abrupt Changes During Generation  )",
            )
            with gr.Accordion('Advanced', open=False):
                res = gr.Slider(
                    minimum=-6,
                    maximum=-2,
                    step=0.1,
                    value=-4.,
                    label="Log 10 Tolerance for Iteration Convergence ( → Faster Convergence, Lower Quality)",
                )
                lr = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=1.,
                    label="Iteration Step Size ( → Faster Convergence)",
                )
                reg_size = gr.Slider(
                    minimum=0.0,
                    maximum=1.,
                    step=0.1,
                    value=0.4,
                    label="Regularization Annealing Speed ( ← Slower, Maybe Easier Convergence)",
                )
                reg_w = gr.Slider(
                    minimum=0.0,
                    maximum=5,
                    step=0.01,
                    value=0.5,
                    label="Regularization Annealing Strength ( ← Stronger Annealing, Slower, Maybe Better Convergence )",
                )
                aa_dim = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=2,
                    label="AA Iteration Memory Size ( → Faster Convergence, Maybe Unstable)",
                )
            with gr.Row():
                checkbox = gr.Checkbox(
                    False,
                    label="Enable"
                ) 
                markdown = gr.Markdown("[How to set parameters? Check our github!](https://github.com/scraed/CharacteristicGuidanceWebUI/tree/main)")
                radio = gr.Radio(
                    choices=["More Prompt", "More ControlNet"], 
                    label="ControlNet Compatible Mode", 
                    value = "More ControlNet"
                    )
            with gr.Blocks() as demo:
                image = gr.Image()
                button = gr.Button("Check Convergence (Please Adjust Regularization If Not Converged)")

                button.click(fn=self.update_plot, outputs=image)
            # with gr.Blocks(show_footer=False) as blocks:
            #        image = gr.Image(show_label=False)
            #        blocks.load(fn=self.update_plot, inputs=None, outputs=image,
            #                        show_progress=False, every=5)

        def get_chg_parameter(key, default=None):
            def get_parameters(d):
                return d.get('CHG', {}).get(key, default)
            return get_parameters

        self.infotext_fields = [
            (checkbox, lambda d: 'CHG' in d),
            (reg_ini, get_chg_parameter('RegS')),
            (reg_range, get_chg_parameter('RegR')),
            (ite, get_chg_parameter('MaxI')),
            (noise_base, get_chg_parameter('NBasis')),
            (chara_decay, get_chg_parameter('Reuse')),
            (res, get_chg_parameter('Tol')),
            (lr, get_chg_parameter('IteSS')),
            (reg_size, get_chg_parameter('ASpeed')),
            (reg_w, get_chg_parameter('AStrength')),
            (aa_dim, get_chg_parameter('AADim')),
            (radio, get_chg_parameter('CMode')),
        ]

        # TODO: add more UI components (cf. https://gradio.app/docs/#components)
        return [reg_ini, reg_range, ite, noise_base, chara_decay, res, lr, reg_size, reg_w, aa_dim, checkbox, markdown, radio]

    def process(self, p, reg_ini, reg_range, ite, noise_base, chara_decay, res, lr, reg_size, reg_w, aa_dim,
                      checkbox, markdown, radio, **kwargs):
        if checkbox:
            # info text will have to be written hear otherwise params.txt will not have the infotext of CHG
            # write parameters to extra_generation_params["CHG"] as json dict with double and single quotes swapped
            parameters = {
                'RegS': reg_ini,
                'RegR': reg_range,
                'MaxI': ite,
                'NBasis': noise_base,
                'Reuse': chara_decay,
                'Tol': res,
                'IteSS': lr,
                'ASpeed': reg_size,
                'AStrength': reg_w,
                'AADim': aa_dim,
                'CMode': radio
            }
            p.extra_generation_params["CHG"] = json.dumps(parameters).translate(quote_swap)
            print("Characteristic Guidance parameters registered")

    # Extension main process
    # Type: (StableDiffusionProcessing, List<UI>) -> (Processed)
    # args is [StableDiffusionProcessing, UI1, UI2, ...]
    def process_batch(self, p, reg_ini, reg_range, ite, noise_base, chara_decay, res, lr, reg_size, reg_w, aa_dim,
                      checkbox, markdown, radio, **kwargs):
        def modified_sample(sample):
            def wrapper(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
                # modules = sys.modules
                if checkbox:
                    # from ssd_samplers_chg_denoiser import CFGDenoiser as CHGDenoiser
                    print("Characteristic Guidance injecting the CFGDenoiser")
                    original_forward = CFGDenoiser.forward
                    CFGDenoiser.forward = CHGDenoiser.forward
                    CFGDenoiser.Chara_iteration = CHGDenoiser.Chara_iteration
                    CFGDenoiser.res_thres = 10 ** res
                    CFGDenoiser.noise_base = noise_base
                    CFGDenoiser.lr_chara = lr
                    CFGDenoiser.ite = ite
                    CFGDenoiser.reg_size = reg_size
                    if reg_ini<=5:
                        CFGDenoiser.reg_ini = reg_ini
                    else:
                        k = 0.8898
                        CFGDenoiser.reg_ini = np.exp(k*(reg_ini-5))/np.exp(0)/k + 5 - 1/k
                    if reg_range<=5:
                        CFGDenoiser.reg_range = reg_range
                    else:
                        k = 0.8898
                        CFGDenoiser.reg_range = np.exp(k*(reg_range-5))/np.exp(0)/k + 5 - 1/k
                    CFGDenoiser.reg_w = reg_w
                    CFGDenoiser.ite_infos = [[], [], []]
                    CFGDenoiser.dxs_buffer = None
                    CFGDenoiser.abt_buffer = None
                    CFGDenoiser.aa_dim = aa_dim
                    CFGDenoiser.chara_decay = chara_decay
                    CFGDenoiser.process_p = p
                    CFGDenoiser.radio_controlnet = radio
                    # CFGDenoiser.CFGdecayS = CFGdecayS
                    try:
                        print("Characteristic Guidance sampling:")
                        result = sample(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength,
                                        prompts)

                    except Exception as e:
                        raise e
                    finally:
                        print("Characteristic Guidance recorded iterations info for " + str(len(CFGDenoiser.ite_infos[0])) + " steps"  )
                        print("Characteristic Guidance recovering the CFGDenoiser")
                        CFGDenoiser.forward = original_forward
                        del CFGDenoiser.Chara_iteration
                        # del CFGDenoiser.res_thres
                        del CFGDenoiser.noise_base
                        del CFGDenoiser.lr_chara
                        del CFGDenoiser.ite
                        del CFGDenoiser.reg_size
                        del CFGDenoiser.reg_ini
                        del CFGDenoiser.reg_range
                        del CFGDenoiser.reg_w
                        del CFGDenoiser.dxs_buffer
                        del CFGDenoiser.abt_buffer
                        del CFGDenoiser.aa_dim
                        del CFGDenoiser.chara_decay
                        del CFGDenoiser.process_p
                        del CFGDenoiser.radio_controlnet
                        
                        
                        # del CFGDenoiser.CFGdecayS
                else:
                    result = sample(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength,
                                    prompts)
                return result

            return wrapper

        # TODO: get UI info through UI object angle, checkbox
        if checkbox:
            print("Characteristic Guidance enabled, warpping the sample method")
            p.sample = modified_sample(p.sample).__get__(p)

        # print(p.sampler_name)
        # TODO: add image edit process via Processed object proc
