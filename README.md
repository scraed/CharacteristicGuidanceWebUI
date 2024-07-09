# Characteristic Guidance Web UI (enhanced sampling for high CFG scale)

## About
Characteristic Guidance Web UI is an extension of for the Stable Diffusion web UI (AUTOMATIC1111). It offers a theory-backed guidance sampling method with improved sample and control quality at high CFG scale (10-30). 

This is the official implementation of [Characteristic Guidance: Non-linear Correction for Diffusion Model at Large Guidance Scale](https://arxiv.org/abs/2312.07586). We are happy to announce that this work has been accepted by ICML 2024.

## News

We release the Turbo version of characteristic guidance. 
  - Easier to use: much fewer parameters to choose
  - Speed optimization (~2x faster)
  - Stability improvement (better convergence at initial steps)
    
Please try the Turbo version by swiching to the branch Turbo_dev. （If you are forge user, please use Karras schedule or unipic to avoid artifacts）

## Features
Characteristic guidance offers improved sample generation and control at high CFG scale. Try characteristic guidance for
- Detail refinement
- Fixing quality issues, like
  - Weird colors and styles
  - Bad anatomy (not guaranteed :rofl:, works better on Stable Diffusion XL)
  - Strange backgrounds
    
Characteristic guidance is compatible with every existing sampling methods in Stable Diffusion WebUI. It now have preliminary support for **Forge UI** and ControlNet.
![1girl running mountain grass](https://github.com/scraed/CharacteristicGuidance/blob/master/static/images/1girl%2C%20running%2C%20mountain%2C%20grass.jpg?raw=true) 
![newspaper news english](https://github.com/scraed/CharacteristicGuidance/blob/master/static/images/newspaper%20news%20english.jpg?raw=true)
![1girl, handstand, sports, close_up](https://github.com/scraed/CharacteristicGuidance/blob/master/static/images/1girl%20handstand%20sports%20close_up.jpg?raw=true)
![StrawberryPancake](https://github.com/scraed/CharacteristicGuidance/blob/master/static/images/StrawberryPancake.jpg?raw=true)
![1girl, kimono](https://scraed.github.io/CharacteristicGuidance/static/images/1girl%20kimono.jpg?raw=true)

For more information and previews, please visit our project website: [Characteristic Guidance Project Website](https://scraed.github.io/CharacteristicGuidance/). 

Q&A: What's the difference with [Dynamical Thresholding](https://github.com/mcmonkeyprojects/sd-dynamic-thresholding)?

They are distinct and independent methods, can be used either independently or in conjunction.

- **Characteristic Guidance**: Corrects both context and color, works at the given CFG scale, iteratively corrects **input** of the U-net according to the Fokker-Planck equation. 
- **Dynamical Thresholding**:  Mainly focusing on color, works to mimic lower CFG scales, clips and rescales **output** of the U-net.

Using [Characteristic Guidance](#) and Dynamical Thresholding simutaneously may further reduce saturation.

![1girl_handstand_sportswear_gym](https://github.com/scraed/CharacteristicGuidance/blob/master/static/images/1girl_handstand_sportswear_gym.jpg?raw=true) 

## Prerequisites
Before installing and using the Characteristic Guidance Web UI, ensure that you have the following prerequisites met:

- **Stable Diffusion WebUI (AUTOMATIC1111)**: Your system must have the [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) by AUTOMATIC1111 installed. This interface is the foundation on which the Characteristic Guidance Web UI operates.
- **Version Requirement**: The extension is developed for Stable Diffusion WebUI **v1.6.0 or higher**. It may works for previous versions but not guaranteed.

## Installation
Follow these steps to install the Characteristic Guidance Web UI extension:

1. Navigate to the "Extensions" tab in the Stable Diffusion web UI.
2. In the "Extensions" tab, select the "Install from URL" option.
3. Enter the URL `https://github.com/scraed/CharacteristicGuidanceWebUI.git` into the "URL for extension's git repository" field.
4. Click on the "Install" button.
5. After waiting for several seconds, a confirmation message should appear indicating successful installation: "Installed into stable-diffusion-webui\extensions\CharacteristicGuidanceWebUI. Use the Installed tab to restart".
6. Proceed to the "Installed" tab. Here, click "Check for updates", followed by "Apply and restart UI" for the changes to take effect. Note: Use these buttons for future updates to the CharacteristicGuidanceWebUI as well.

## Usage
The Characteristic Guidance Web UI features an interactive interface for both txt2img and img2img mode. 
![Gradio UI for CharacteristicGuidanceWebUI](https://github.com/scraed/CharacteristicGuidanceWebUI/blob/main/CHGextension_pic.PNG?raw=true)

**The characteristic guidance is slow compared to classifier-free guidance. We recommend the user to generate image with classifier-free guidance at first, then try characteristic guidance with the same prompt and seed to enhance the image.**

### Activation
- `Enable` Checkbox: Toggles the activation of the Characteristic Guidance features.

### Visualization and Testing
- `Check Convergence` Button: Allows users to test and visualize the convergence of their settings. Adjust the regularization parameters if the convergence is not satisfactory.

In practice, convergence is not always guaranteed. **If characteristic guidance fails to converge at a certain time step, classifier-free guidance will be adopted at that time step**. 

Below are the parameters you can adjust to customize the behavior of the guidance correction:

### Basic Parameters
- `Regularization Strength`: Range 0.0 to 10.0 (default: 1). Adjusts the strength of regularization at the beginning of sampling, larger regularization means easier convergence and closer alignment with CFG (Classifier Free Guidance).
- `Regularization Range Over Time`: Range 0.01 to 10.0 (default: 1). Modifies the range of time being regularized, larger time means slow decay in regularization strength hence more time steps being regularized, affecting convergence difficulty and the extent of correction.
- `Max Num. Characteristic Iteration`: Range 1 to 50 (default: 50). Determines the maximum number of characteristic iterations per sampling time step.
- `Num. Basis for Correction`: Range 0 to 10 (default: 0). Sets the number of bases for correction, influencing the amount of correction and convergence behavior. More basis means better quality but harder convergence. Basis number = 0 means batch-wise correction, > 0 means channel-wise correction. 
- `CHG Start Step`: Range 0 to 0.25 (default: 0). Characteristic guidance begins to influence the process from the specified percentage of steps, indicated by `CHG Start Step`.
- `CHG End Step`: Range 0.25 to 1 (default: 0). Characteristic guidance ceases to have an effect from the specified percentage of steps, denoted by `CHG End Step`. Setting this value to approximately 0.4 can significantly speed up the generation process without substantially altering the outcome.
- `ControlNet Compatible Mode`
  - `More Prompt`: Controlnet is turned off when iteratively solving characteristic guidance correction.
  - `More ControlNet`: Controlnet is turned on when iteratively solving characteristic guidance correction.

### Advanced Parameters
- `Reuse Correction of Previous Iteration`: Range 0.0 to 1.0 (default: 1.0). Controls the reuse of correction from previous iterations to reduce abrupt changes during generation.
- `Log 10 Tolerance for Iteration Convergence`: Range -6 to -2 (default: -4). Adjusts the tolerance for iteration convergence, trading off between speed and image quality.
- `Iteration Step Size`: Range 0 to 1 (default: 1.0). Sets the step size for each iteration, affecting the speed of convergence.
- `Regularization Annealing Speed`: Range 0.0 to 1.0 (default: 0.4). How fast the regularization strength decay to desired rate. Smaller value potentially easing convergence.
- `Regularization Annealing Strength`: Range 0.0 to 5 (default: 0.5). Determines the how important regularization annealing is in characteristic guidance interations. Higher value means higher priority to bring regularization level to specified regularization strength. Affecting the balance between annealing and convergence.
- `AA Iteration Memory Size`: Range 1 to 10 (default: 2). Specifies the memory size for AA (Anderson Acceleration) iterations, influencing convergence speed and stability.


Please experiment with different settings, especially **regularization strength and time range**, to achieve better convergence for your specific use case. (According to my experience, high CFG scale need relatively large regularization strength and time range for convergence, while low CFG scale prefers lower regularization strength and time range for more guidance correction.)

### How to Set Parameters (Preliminary Guide)
Here is my recommended approach for parameter setting:

1. Start by running characteristic guidance with the default parameters (Use `Regularization Strength`=5 for Stable Diffusion XL).
2. Verify convergence by clicking the `Check Convergence` button.
3. If convergence is achieved easily:
   - Decrease the `Regularization Strength` and `Regularization Range Over Time` to enhance correction.
   - If the `Regularization Strength` is already minimal, consider increasing the `Num. Basis for Correction` for improved performance.
4. If convergence is not reached:
   - Increment the `Max Num. Characteristic Iteration` to allow for additional iterations.
   - Should convergence still not occur, raise the `Regularization Strength` and `Regularization Range Over Time` for increased regularization.



## Updates
### July 9, 2024: Release Turbo_dev branch.
- New technique to stablize the iteration at beginning steps
- Avoid redundant iteration steps to accelerate generation. 


### June 24, 2024: Preliminary Support for Forge.
- **Thanks to our team member [@charrywhite](https://github.com/charrywhite)**: The UI now have preliminary supports [forge](https://github.com/lllyasviel/stable-diffusion-webui-forge).
- Fixed minor incompatibility with WebUI v1.9


### February 3, 2024: New parameters accelerating the generation.
- **Thanks to [@v0xie](https://github.com/v0xie)**: The UI now supports two more parameters.
- `CHG Start Step`: Range 0 to 0.25 (default: 0). Characteristic guidance begins to influence the process from the specified percentage of steps, indicated by `CHG Start Step`.
- `CHG End Step`: Range 0.25 to 1 (default: 0). Characteristic guidance ceases to have an effect from the specified percentage of steps, denoted by `CHG End Step`. Setting this value to approximately 0.4 can significantly **speed up** the generation process without substantially altering the outcome.

### January 28, 2024: Modify how parameter `Reuse Correction of Previous Iteration` works
- **Effect**: Move parameter `Reuse Correction of Previous Iteration` to advanced parameters. Its default value is set to 1 to accelerate convergence. It is now using the same update direction as the case `Reuse Correction of Previous Iteration` = 0 regardless of its value. 
- **User Action Required**: Please delete "ui-config.json" from the stable diffusion WebUI root directory for the update to take effect.
- **Issue**: Infotext with `Reuse Correction of Previous Iteration` > 0 may not generate the same image as previous version.

### January 28, 2024: Allow Num. Basis for Correction = 0
- **Effect**: Now the Num. Basis for Correction can takes value 0 which means batch-wise correction instead of channel-wise correction. It is a more suitable default value since it converges faster.
- **User Action Required**: Please delete "ui-config.json" from the stable diffusion WebUI root directory for the update to take effect.

### January 14, 2024: Bug fix: allow prompts with more than 75 tokens
- **Effect**: Now the extension still works if the prompt have more than 75 tokens.

### January 13, 2024: Add support for V-Prediction model 
- **Effect**: Now the extension supports models trained in V-prediction mode.

### January 12, 2024: Add support for 'AND' prompt combination 
- **Effect**: Now the extension supports the 'AND' word in positive prompt.
- **Current Limitations**: Note that characteristic guidance only give correction between positive and negative prompt. Therefore positive prompts combined by 'AND' will be averaged when computing the correction.

### January 8, 2024: Improved Guidance Settings
- **Extended Settings Range**: `Regularization Strength` & `Regularization Range Over Time` can now go up to 10.
- **Effect**: Reproduce classifier-free guidance results at high values of `Regularization Strength` & `Regularization Range Over Time`.
- **User Action Required**: Please delete "ui-config.json" from the stable diffusion WebUI root directory for the update to take effect.

### January 6, 2024: Integration of ControlNet
- **Early Support**: We're excited to announce preliminary support for ControlNet.
- **Current Limitations**: As this is an early stage, expect some developmental issues. The integration of ControlNet and characteristic guidance remains a scientific open problem (which I am investigating). Known issues include:
   - Iterations failing to converge when ControlNet is in reference mode.

### January 3, 2024: UI Enhancement for Infotext
- **Thanks to [@w-e-w](https://github.com/w-e-w)**: The UI now supports infotext reading.
- **How to Use**: Check out this [PR](https://github.com/scraed/CharacteristicGuidanceWebUI/pull/1) for detailed instructions.


## Compatibility and Issues

### July 9, 2024: Bad Turbo_dev branch output on Forge.
- Sometimes the generated images has wierd artifacts on Forge Turbo_dev. Please use Karras schedule to avoid it.

### June 24, 2024: Inconsistent Forge Implementation.
- Note that the current forge implementation of CHG does not always generate the same image as CHG on WebUI. See this [pull request](https://github.com/scraed/CharacteristicGuidanceWebUI/pull/13). We are still investigating why it happends.


## Citation
If you utilize characteristic guidance in your research or projects, please consider citing our paper:
```bibtex
@misc{zheng2023characteristic,
      title={Characteristic Guidance: Non-linear Correction for DDPM at Large Guidance Scale},
      author={Candi Zheng and Yuan Lan},
      year={2023},
      eprint={2312.07586},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}


