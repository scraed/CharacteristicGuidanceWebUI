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

def CHGdenoiserConstruct():
    CHGDenoiserStr = '''
class CHGDenoiser(CFGDenoiser):
    def __init__(self, sampler):
        super().__init__(sampler)

'''
    CHGDenoiserStr += inspect.getsource(CFGDenoiser.forward)

    hijack_breakpoints = ["\s+if shared.opts.batch_cond_uncond:", "\s+denoised_image_indexes = \[x\[0\]\[0\] for x in conds_list\]"]


    breakpoint1 = [m.start() for m in re.finditer(hijack_breakpoints[0], CHGDenoiserStr)]
    breakpoint2 = [m.start() for m in re.finditer(hijack_breakpoints[1], CHGDenoiserStr)]
    if len(breakpoint1) != 1 or len(breakpoint2) != 1:
        print( "Characteristic Guidance detected multiple hijack break point. This is caused by version incompatibility. Please report this issue." )
    bp1 = breakpoint1[0]
    bp2 = breakpoint2[0]
    CHGDenoiserStr1 = CHGDenoiserStr[:bp1]
    CHGDenoiserStr2 = CHGDenoiserStr[bp1:bp2]
    CHGDenoiserStr3 = CHGDenoiserStr[bp2:]
    CHGhijackStr = """

            if shared.opts.batch_cond_uncond:
                x_out = Chara_iteration(self, None, x_in, sigma_in, tensor, uncond, cond_scale, image_cond_in, is_edit_model, skip_uncond, make_condition_dict, shared.opts.batch_cond_uncond, batch_size, cond_in=cond_in, x_out = None)
            else:
                x_out = Chara_iteration(self, None, x_in, sigma_in, tensor, uncond, cond_scale, image_cond_in, is_edit_model, skip_uncond, make_condition_dict, shared.opts.batch_cond_uncond, batch_size, cond_in=cond_in, x_out = None)
        else:
            x_out = Chara_iteration(self, None, x_in, sigma_in, tensor, uncond, cond_scale, image_cond_in, is_edit_model, skip_uncond, make_condition_dict, shared.opts.batch_cond_uncond, batch_size, cond_in=None, x_out = None)

    """
    CHGDenoiserStr = CHGDenoiserStr1+CHGhijackStr+CHGDenoiserStr3
    return CHGDenoiserStr