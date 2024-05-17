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
from modules_forge import forge_sampler

from ldm_patched.modules.conds import CONDRegular, CONDCrossAttn
from ldm_patched.modules import model_management
from ldm_patched.modules.ops import cleanup_cache
from ldm_patched.modules.samplers import *
from modules_forge.forge_sampler import *

print("**********Read forge sample code *********")

def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
        edit_strength = sum((item['strength'] if 'strength' in item else 1) for item in cond)
        if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
            uncond_ = None
        else:
            uncond_ = uncond

        for fn in model_options.get("sampler_pre_cfg_function", []):
            model, cond, uncond_, x, timestep, model_options = fn(model, cond, uncond_, x, timestep, model_options)
        #this is where neural network evaluation happends at x
        cond_pred, uncond_pred = calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)
        
        print( "Model type: ",model.model_type )
        print("cond_pred", cond_pred.shape)
        print("uncond_pred",cond_pred.shape)

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
    uncond = cond_from_a1111_to_patched_ldm(denoiser_params.text_uncond)
    cond = cond_from_a1111_to_patched_ldm_weighted(denoiser_params.text_cond, cond_composition)
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

    denoised = sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)
    return denoised


def CHGdenoiserConstruct():
    CHGDenoiserStr = '''
class CHGDenoiser(CFGDenoiser):
    def __init__(self, sampler):
        super().__init__(sampler)
    def forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
        if state.interrupted or state.skipped:
            raise sd_samplers_common.InterruptedException

        original_x_device = x.device
        original_x_dtype = x.dtype

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

        denoised = forge_inject.forge_sample(self, denoiser_params=denoiser_params,
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