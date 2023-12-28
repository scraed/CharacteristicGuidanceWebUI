# Characteristic Guidance Web UI

## About
Characteristic Guidance Web UI is a tool that offers large CFG scale correction for the Stable Diffusion web UI (AUTOMATIC1111). This project aims at enhancing the sampling and control quality of diffusion models at larger CFG guidance scale.

## Features
- Improved sample generation control at CFG scale
- Compatible with existing sampling methods

For a detailed overview of features and previews, please visit our project website: [Characteristic Guidance Project Website](https://scraed.github.io/CharacteristicGuidance/). 

## Installation
Follow these steps to install the CharacteristicGuidanceWebUI extension:

1. Navigate to the "Extensions" tab in the Stable Diffusion web UI.
2. In the "Extensions" tab, select the "Install from URL" option.
3. Enter the URL `https://github.com/scraed/CharacteristicGuidanceWebUI.git` into the "URL for extension's git repository" field.
4. Click on the "Install" button.
5. After waiting for several seconds, a confirmation message should appear indicating successful installation: "Installed into stable-diffusion-webui\extensions\CharacteristicGuidanceWebUI. Use the Installed tab to restart".
6. Proceed to the "Installed" tab. Here, click "Check for updates", followed by "Apply and restart UI" for the changes to take effect. Note: Use these buttons for future updates to the CharacteristicGuidanceWebUI as well.

## Usage
The CharacteristicGuidanceWebUI features an interactive interface built with Gradio. Below are the parameters you can adjust to customize the behavior of the guidance correction:

### Basic Parameters
- **Regularization Strength**: Range 0.0 to 10.0 (default: 1). Adjusts the strength of regularization, facilitating easier convergence and closer alignment with CFG (Classifier Free Guidance).
- **Regularization Range Over Time**: Range 0.01 to 5.0 (default: 1). Modifies the regularization range over time, affecting convergence difficulty and the extent of correction.
- **Max Num. Characteristic Iteration**: Range 1 to 50 (default: 30). Determines the maximum number of characteristic iterations, balancing speed and convergence quality.
- **Num. Basis for Correction**: Range 1 to 6 (default: 1). Sets the number of bases for correction, influencing the amount of correction and convergence behavior.
- **Reuse Correction of Previous Iteration**: Range 0.0 to 1.0 (default: 0.0). Controls the reuse of correction from previous iterations to reduce abrupt changes during generation.

### Advanced Parameters
- **Log 10 Tolerance for Iteration Convergence**: Range -6 to -2 (default: -4). Adjusts the tolerance for iteration convergence, trading off between speed and image quality.
- **Iteration Step Size**: Range 0 to 1 (default: 1.0). Sets the step size for each iteration, affecting the speed of convergence.
- **Regularization Annealing Speed**: Range 0.0 to 1.0 (default: 0.4). Controls the speed of regularization annealing, potentially easing convergence.
- **Regularization Annealing Strength**: Range 0.0 to 5 (default: 0.5). Determines the strength of regularization annealing, affecting the balance between annealing strength and convergence.
- **AA Iteration Memory Size**: Range 1 to 10 (default: 2). Specifies the memory size for AA (Anderson Acceleration) iterations, influencing convergence speed and stability.

### Activation
- **Enable Checkbox**: Toggles the activation of the Characteristic Guidance features.

### Visualization and Testing
- **Check Convergence Button**: Allows users to test and visualize the convergence of their settings. Adjust the regularization parameters if the convergence is not satisfactory.

These parameters provide extensive control over the behavior and performance of the diffusion model, allowing for fine-tuning to achieve desired results. Experiment with different settings to find the optimal balance for your specific use case.

## Examples
Show some examples of how your tool improves the performance of diffusion models. Include images or links to results, if available.

## Contributing
Guidelines for how others can contribute to your project. Include instructions for submitting pull requests, coding standards, and how to report bugs.

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


