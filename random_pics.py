import random
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


subjects = [
    "a cyberpunk samurai", "an alien creature", "a futuristic city",
    "a medieval knight", "a robot musician", "a cosmic dragon",
    "a surreal landscape", "a floating castle", "a galaxy-sized whale"
]

styles = [
    "in anime style", "in pixel art", "in hyperrealism",
    "in watercolor", "as a digital painting", "in vaporwave colors",
    "in minimalism", "as a cinematic shot", "in surrealism"
]

locations = [
    "on Mars", "in the middle of Times Square", "underwater city",
    "inside a dream", "in a neon jungle", "floating in space",
    "in an ancient temple", "on a distant exoplanet", "in virtual reality"
]

vibes = [
    "glowing with mysticism", "surrounded by neon lights", "covered in fog",
    "with dramatic lighting", "floating in zero gravity", "bathed in starlight",
    "with abstract fractals", "with cosmic energy", "in apocalyptic vibes"
]

def generate_prompt():
    return f"{random.choice(subjects)} {random.choice(styles)} {random.choice(locations)}, {random.choice(vibes)}"


num_prompts = 3
generated_prompts = [generate_prompt() for _ in range(num_prompts)]

print("Generated Prompts:")
for p in generated_prompts:
    print(" -", p)


results = pipeline(
    prompt=generated_prompts,
    num_inference_steps=60,
    guidance_scale=4.5,
    max_sequence_length=512,
)

images = results.images

for i, img in enumerate(images):
    img.save(f"image_{i}_randomprompt.png")

print(f"{len(images)} random images saved")
