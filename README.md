# SD-image-generator
This is the full source code for the video! It includes all the examples, key differences, and is super easy to customize. Hope you enjoyed the video — feel free to fork, tweak, and build your own version!

## Overview
This repository contains Python scripts using the Stable Diffusion 3.5 Medium model via HuggingFace Diffusers, focused on generating high-quality AI images. The scripts provide batch inference, random prompt generation for creative outputs, and a tester stub for future development.

## Files Included
- main.py: Batch generation with custom prompts for advanced creativity.
- random_pics.py: Automated random prompt construction and image synthesis for diverse visuals.
- tester.py: Template for future extensions or testing utility.

## Installation
Clone the Repository:















Python Environment:
- Python 3.9 or higher is recommended.
- Set up a virtual environment for isolation:
- For CUDA support, ensure that your system has the correct NVIDIA drivers and CUDA toolkit.

## Usage Instructions

### main.py
Purpose: Generate a batch of images from a list of detailed prompts using Stable Diffusion 3.5 Medium.

Run:
python3 main.py

Output: Images are saved as image_0_high_inference.png, image_1_high_inference.png, etc., in the working directory.

Editing Prompts: Modify the `prompts` list in the source code to generate your desired images.

Tweaking Parameters:
- num_inference_steps: Increase for finer detail but slower speed.
- guidance_scale: Adjust for more/less adherence to prompts.
- max_sequence_length: Set to control prompt length (Max is 512).

### random_pics.py
Purpose: Create randomized creative prompts for image generation automatically.

Run:
python3 random_pics.py

Customization:
- Change any of the `subjects`, `styles`, `locations`, or `vibes` lists for custom flavor and output diversity.
- Modify `num_prompts` for generating more or fewer images per run.

Preview Prompts: The script prints generated prompts for transparency and further manual reuse.

### tester.py
Template file: Use this as a starting point for testing your own models, adding debug prints, or integrating further functionality.

## Customization
- For model variations, change the `model_id` to any supported by HuggingFace Diffusers.
- To alter quantization, modify the parameters of `BitsAndBytesConfig` as needed for different performance profiles.
- Experiment with batch sizes and CPU/GPU settings (`pipeline.enable_model_cpu_offload()`) for hardware adaptation.

## Downloading Images
- All generated images are automatically stored in the current working directory with descriptive filenames.
- To change location, update the path in the `img.save()` calls.
- Images are standard PNG format for maximum compatibility.

## Troubleshooting
- CUDA errors: Ensure the correct version of PyTorch and CUDA is installed for your GPU.
- Dependency issues: Upgrade packages with `pip install --upgrade diffusers torch transformers`.
- File not found: Check file paths and directory context if invoking from other locations.
- If you face more problems try to debug it using chatgpt for version errors and more if you are not that familiar with python or coding in general
