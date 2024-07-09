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
from extensions.CharacteristicGuidanceWebUI.scripts.CharaIte import Chara_iteration

try:
    from modules_forge import forge_sampler
    isForge = True
except Exception:
    isForge = False

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


if not isForge:
    from extensions.CharacteristicGuidanceWebUI.scripts.webui_CHG import CHGdenoiserConstruct
    exec( CHGdenoiserConstruct() )
else:
    from extensions.CharacteristicGuidanceWebUI.scripts.forge_CHG import CHGdenoiserConstruct
    import extensions.CharacteristicGuidanceWebUI.scripts.forge_CHG as forge_CHG
    exec( CHGdenoiserConstruct() )



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
            reg_ini = CFGDenoiser.reg_ini
            reg_range = CFGDenoiser.reg_range 
            noise_base = CFGDenoiser.noise_base
            start_step = CFGDenoiser.chg_start_step
        except:
            res_thres = 0.1
            reg_ini=1
            reg_range=1
            noise_base = 1
            start_step = 0
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Converged'),
                           Patch(facecolor='yellow', label='Barely Converged'),
                           Patch(facecolor='red', label='Not Converged')]
        def get_title(reg_ini, reg_range, noise_base, num_no_converge, pos_no_converge):
            title = ""
            prompts = ["Nice! All iterations converged.\n ",
            "Lowering the regularization strength may be better.\n ",
            "One iteration not converge, but it is OK.\n ",
            "Two or more iteration not converge, maybe you should increase regularization strength.\n ",
            "Steps in the middle didn't converge, maybe you should increase regularization time range.\n ",
            "The regularization strength is already small. Increasing the number of basis worth a try.\n ",
            "If you think context changed too much, increase the regularization strength. \n ",
            "Increase the regularization strength may be better.\n ",
            "If you think context changed too little, lower the regularization strength. \n ",
            "If you think context changed too little, lower the regularization time range. \n ",
            "Number of Basis maybe too high, try lowering it. \n "
            ]
            if num_no_converge <=0:
                title += prompts[0]
            if num_no_converge <=0 and reg_ini > 0.5:
                title += prompts[1]
            if num_no_converge == 1:
                title += prompts[2]
                title += prompts[7]
            if num_no_converge >1:
                title += prompts[3]
                title += prompts[7]
            if pos_no_converge > 0.3:
                title += prompts[4]
            if num_no_converge <=0 and reg_ini <= 0.5:
                title += prompts[5]
            if num_no_converge <=0 and reg_ini < 5:
                title += prompts[6]
            if num_no_converge <=0 and reg_ini >= 5:
                title += prompts[8]
                title += prompts[9]
            if num_no_converge >=2 and noise_base >2:
                title += prompts[10]
            alltitles = title.split("\n")[:-1]
            n = np.random.randint(len(alltitles))
            return alltitles[n]
        # Create bar plot
        fig, axs = plt.subplots(len(res), 1, figsize=(10, 4.5 * len(res)))
        if len(res) > 1:
            # Example plotting code
            for i in range(len(res)):
                num_no_converge = 0
                pos_no_converge = 0
                for j, r in enumerate(res[i]):
                    if r >= res_thres:
                        num_no_converge+=1
                        pos_no_converge = max(j,pos_no_converge)
                pos_no_converge = pos_no_converge/(len(res[i])+1)
                # Categorize each result and assign colors
                colors = ['green' if r < res_thres else 'yellow' if r < 10 * res_thres else 'red' for r in res[i]]
                axs[i].bar(np.arange(len(ite_num[i]))+start_step, ite_num[i], color=colors)
                # Create legend
                axs[i].legend(handles=legend_elements, loc='upper right')

                # Add labels and title
                axs[i].set_xlabel('Diffusion Step')
                axs[i].set_ylabel('Num. Characteristic Iteration')
                ax2 = axs[i].twinx()
                ax2.plot(np.arange(len(ite_num[i]))+start_step, reg[i], linewidth=4, color='C1', label='Regularization Level')
                ax2.set_ylabel('Regularization Level')
                ax2.set_ylim(bottom=0.)
                ax2.legend(loc='upper left')
                title = get_title(reg_ini, reg_range, noise_base, num_no_converge, pos_no_converge)
                ax2.set_title(title)
                ax2.autoscale()
            # axs[i].set_title('Convergence Status of Iterations for Each Step')
        elif len(res) == 1:
            num_no_converge = 0
            pos_no_converge = 0
            for j, r in enumerate(res[0]):
                if r >= res_thres:
                    num_no_converge+=1
                    pos_no_converge = max(j,pos_no_converge)
            pos_no_converge = pos_no_converge/(len(res[0])+1)
            colors = ['green' if r < res_thres else 'yellow' if r < 10 * res_thres else 'red' for r in res[0]]
            axs.bar(np.arange(len(ite_num[0]))+start_step, ite_num[0], color=colors)
            # Create legend
            axs.legend(handles=legend_elements, loc='upper right')

            # Add labels and title
            axs.set_xlabel('Diffusion Step')
            axs.set_ylabel('Num. Characteristic Iteration')
            ax2 = axs.twinx()
            title = get_title(reg_ini, reg_range, noise_base, num_no_converge, pos_no_converge)
            ax2.plot(np.arange(len(ite_num[0]))+start_step, reg[0], linewidth=4, color='C1', label='Regularization Level')
            ax2.set_ylabel('Regularization Level')
            ax2.set_ylim(bottom=0.)
            ax2.legend(loc='upper left')
            ax2.set_title(title)
            ax2.autoscale()
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
        with gr.Accordion('Characteristic Guidance (CHG)', open=False):
            log_alpha_reg = gr.Slider(
                minimum=-2.,
                maximum=3.,
                step=0.1,
                value=-1.,
                label="Regularization ( → Easier Convergence, Closer to Classfier-Free. Please try various values)",
            )
            ite = gr.Slider(
                minimum=1,
                maximum=50,
                step=1,
                value=50,
                label="Max Num. Characteristic Iteration ( → Slow but Better Convergence)",
            )
            noise_base = gr.Slider(
                minimum=0,
                maximum=10,
                step=1,
                value=0,
                label="Num. Basis for Correction ( ← Less Correction, Better Convergence)",
            )
            guide_time = gr.Slider(
                minimum=0.0,
                maximum=5.,
                step=0.01,
                value=0.,
                label="The guidance applied time ( ← Slower, More Correction.)",
            )
            with gr.Accordion('Advanced', open=False):
                reg_ini = gr.Slider(
                    minimum=0.0,
                    maximum=10.,
                    step=0.1,
                    value=0.0,
                    label="Legacy Regularization Strength ( → Easier Convergence, Closer to Classfier-Free. Please try various values)",
                )
                reg_range = gr.Slider(
                    minimum=0.01,
                    maximum=10.,
                    step=0.01,
                    value=0.01,
                    label="Legacy Regularization Range Over Time ( ← Harder Convergence, More Correction. Please try various values)",
                )
                with gr.Row(open=True):
                    start_step = gr.Slider(
                        minimum=0.0,
                        maximum=0.25,
                        step=0.01,
                        value=0.0,
                        label="CHG Start Step ( Use CFG before Percent of Steps. )",
                    )
                    stop_step = gr.Slider(
                        minimum=0.25,
                        maximum=1.0,
                        step=0.01,
                        value=1.0,
                        label="CHG End Step ( Use CFG after Percent of Steps. )",
                    )
                chara_decay = gr.Slider(
                    minimum=0.,
                    maximum=1.,
                    step=0.01,
                    value=1.,
                    label="Reuse Correction of Previous Iteration ( → Suppress Abrupt Changes During Generation  )",
                )
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
                button = gr.Button("Check Convergence (Please Adjust Regularization Strength & Range Over Time If Not Converged)")

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
            (start_step, get_chg_parameter('StartStep')),
            (stop_step, get_chg_parameter('StopStep')),
            (log_alpha_reg, get_chg_parameter('RegA')),
            (guide_time, get_chg_parameter('GuideT'))
        ]

        # TODO: add more UI components (cf. https://gradio.app/docs/#components)
        return [log_alpha_reg, reg_ini, reg_range, ite, noise_base, chara_decay, res, lr, reg_size, reg_w, aa_dim, checkbox, markdown, radio, start_step, stop_step, guide_time]

    def process(self, p, log_alpha_reg, reg_ini, reg_range, ite, noise_base, chara_decay, res, lr, reg_size, reg_w, aa_dim,
                      checkbox, markdown, radio, start_step, stop_step, guide_time, **kwargs):
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
                'CMode': radio,
                'StartStep': start_step,
                'StopStep': stop_step,
                'RegA': log_alpha_reg,
                'GuideT': guide_time
            }
            p.extra_generation_params["CHG"] = json.dumps(parameters).translate(quote_swap)
            print("Characteristic Guidance parameters registered")

    # Extension main process
    # Type: (StableDiffusionProcessing, List<UI>) -> (Processed)
    # args is [StableDiffusionProcessing, UI1, UI2, ...]
    def process_batch(self, p, log_alpha_reg, reg_ini, reg_range, ite, noise_base, chara_decay, res, lr, reg_size, reg_w, aa_dim,
                      checkbox, markdown, radio, start_step, stop_step, guide_time, **kwargs):
        #print('*********process batch*********')
        def modified_sample(sample):
            def wrapper(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
                # modules = sys.modules
                if checkbox:
                    # from ssd_samplers_chg_denoiser import CFGDenoiser as CHGDenoiser
                    print("Characteristic Guidance modifying the CFGDenoiser")


                    original_forward = CFGDenoiser.forward
                    def _call_forward(self, *args, **kwargs):
                        if self.chg_start_step <= self.step < self.chg_stop_step:
                            return CHGDenoiser.forward(self, *args, **kwargs)
                        else:
                            return original_forward(self, *args, **kwargs)
                    CFGDenoiser.forward = _call_forward
                    #CFGDenoiser.Chara_iteration = Chara_iteration
                    print('*********cfg denoiser res thres def ************')
                    CFGDenoiser.res_thres = 10 ** res
                    CFGDenoiser.noise_base = noise_base
                    CFGDenoiser.lr_chara = lr
                    CFGDenoiser.ite = ite
                    CFGDenoiser.guide_time = np.exp(guide_time)
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
                    CFGDenoiser.alpha_reg = 10**log_alpha_reg 
                    CFGDenoiser.aa_dim = aa_dim
                    CFGDenoiser.chara_decay = chara_decay
                    CFGDenoiser.process_p = p
                    CFGDenoiser.radio_controlnet = radio
                    constrain_step = lambda total_step, step_pct: max(0, min(round(total_step * step_pct), total_step))
                    CFGDenoiser.chg_start_step = constrain_step(p.steps, start_step)
                    CFGDenoiser.chg_stop_step = constrain_step(p.steps, stop_step)
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

