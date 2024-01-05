# Characteristic Guidance Web UI (High CFG scale fixed sampling)

## About
Characteristic Guidance Web UI is a tool that offers theroy-backed high CFG scale correction for the Stable Diffusion web UI (AUTOMATIC1111), aims at enhancing the sampling and control quality of diffusion models at large CFG guidance scale.

## Features
![newspaper news english](https://github.com/scraed/CharacteristicGuidance/blob/master/static/images/newspaper%20news%20english.jpg?raw=true)
![1girl, handstand, sports, close_up](https://github.com/scraed/CharacteristicGuidance/blob/master/static/images/1girl%20handstand%20sports%20close_up.jpg?raw=true)
![StrawberryPancake](https://github.com/scraed/CharacteristicGuidance/blob/master/static/images/StrawberryPancake.jpg?raw=true)

- Improved sample generation control at high CFG scale
- Compatible with existing sampling methods

For more information and previews, please visit our project website: [Characteristic Guidance Project Website](https://scraed.github.io/CharacteristicGuidance/). 

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

The characteristic guidance is slow compared to classifier-free guidance. We recommend the user to generate image with classifier-free guidance at first, then try characteristic guidance with the same prompt and seed to enhance the image.

Below are the parameters you can adjust to customize the behavior of the guidance correction:

### Basic Parameters
- **Regularization Strength**: Range 0.0 to 5.0 (default: 1). Adjusts the strength of regularization at the beginning of sampling, larger regularization means easier convergence and closer alignment with CFG (Classifier Free Guidance).
- **Regularization Range Over Time**: Range 0.01 to 5.0 (default: 1). Modifies the range of time being regularized, larger time means slow decay in regularization strength hence more time steps being regularized, affecting convergence difficulty and the extent of correction.
- **Max Num. Characteristic Iteration**: Range 1 to 50 (default: 30). Determines the maximum number of characteristic iterations per sampling time step.
- **Num. Basis for Correction**: Range 1 to 6 (default: 1). Sets the number of bases for correction, influencing the amount of correction and convergence behavior. More basis means better quality but harder convergence
- **Reuse Correction of Previous Iteration**: Range 0.0 to 1.0 (default: 0.0). Controls the reuse of correction from previous iterations to reduce abrupt changes during generation. Suppress Abrupt Changes During Generation.

### Advanced Parameters
- **Log 10 Tolerance for Iteration Convergence**: Range -6 to -2 (default: -4). Adjusts the tolerance for iteration convergence, trading off between speed and image quality.
- **Iteration Step Size**: Range 0 to 1 (default: 1.0). Sets the step size for each iteration, affecting the speed of convergence.
- **Regularization Annealing Speed**: Range 0.0 to 1.0 (default: 0.4). Controls the speed of regularization annealing (We set regularization to 5 then let it decay to specified regularization strength, annealing speed determines how fast the decay rate). Smaller speed potentially easing convergence.
- **Regularization Annealing Strength**: Range 0.0 to 5 (default: 0.5). Determines the how important regularization annealing is in characteristic guidance interations. Higher value means higher priority to bring regularization level to specified regularization strength. Affecting the balance between annealing and convergence.
- **AA Iteration Memory Size**: Range 1 to 10 (default: 2). Specifies the memory size for AA (Anderson Acceleration) iterations, influencing convergence speed and stability.

### Activation
- **Enable Checkbox**: Toggles the activation of the Characteristic Guidance features.

### Visualization and Testing
- **Check Convergence Button**: Allows users to test and visualize the convergence of their settings. Adjust the regularization parameters if the convergence is not satisfactory.

In practice, convergence is not always guaranteed. **If characteristic guidance fails to converge at a certain time step, classifier-free guidance will be adopted at that time step**. 

Please experiment with different settings, especially **regularization strength and time range**, to achieve better convergence for your specific use case. (According to my experience, high CFG scale need relatively large regularization strength and time range for convergence, while low CFG scale prefers lower regularization strength and time range for more guidance correction.)

### Compatibility

This extension is presently incompatible with ControlNet (Iteration wouldn't converge if ControlNet is active); however, we are committed to developing a solution in the future.

### Updates

Now the infotext can be read by the UI thanks to [@w-e-w](https://github.com/w-e-w) , see how to use at this [PR](https://github.com/scraed/CharacteristicGuidanceWebUI/pull/1). 

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


