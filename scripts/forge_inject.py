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

import k_diffusion.utils as utils_old
from ldm_patched.modules.conds import CONDRegular, CONDCrossAttn
from ldm_patched.modules import model_management
from ldm_patched.modules.ops import cleanup_cache
from ldm_patched.modules.samplers import *

try:
    from modules_forge import forge_sampler
    from modules_forge.forge_sampler import *
    isForge = True
except Exception:
    isForge = False

from extensions.CharacteristicGuidanceWebUI.scripts.CharaIte import Chara_iteration

# 1st edit by https://github.com/comfyanonymous/ComfyUI
# 2nd edit by Forge Official


print("**********Read forge sample code *********")

def calc_cond_uncond_batch(self,model, cond, uncond, x_in, timestep, model_options,cond_scale):
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
    # print( "Model type: ",model.model_type )
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
    #print(CHGDenoiserStr)
    return CHGDenoiserStr