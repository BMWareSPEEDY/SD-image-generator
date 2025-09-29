from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch

model_id = "stabilityai/stable-diffusion-3.5-medium"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
)

pipeline.enable_model_cpu_offload()

prompts = [
    "Minecraft Steve doing the gritty dance.",
    "A chicken jockey holding a controller playing Call of Duty",
    "Shrek driving a Tesla with sunglasses on.",
    "SpongeBob holding a ‘Subscribe’ sign like a protest."
]

results = pipeline(
    prompt=prompts,
    num_inference_steps=40,
    guidance_scale=8,
    max_sequence_length=512,
)

images = results.images

for i, img in enumerate(images):
    img.save(f"image_{i}_art.png")

print("images saved")