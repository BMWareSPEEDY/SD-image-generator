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
    "In the distance a rock group plays hard. they have banks of speakers behind them, and to each side, and into the foreground are piles of building materials which are being assembled into house by the music buildingbylaushine, sphereglassbuilding,",

    "1 man in hair　Detective Conan clothes　animal ear　London cityscape　anime style　Thick Coating　Professional technology",

    "Hamdan bin Mohammed bin Rashid Al Maktoum skates on the ice in a sports uniform",

    "mammal quadruped alien creatures on an exoplanet",

    "A cosmic field of floating thought-orbs, each orb containing a distant echo, suspended mid-vibration in a silent storm, liquid lines of memory drifting between them, no gravity — abstract psychic dimension, glowing mysticism, SDXL, soft purples and electric blue threads"
]

results = pipeline(
    prompt=prompts,
    num_inference_steps=40,
    guidance_scale=4.5,
    max_sequence_length=512,
)

images = results.images

for i, img in enumerate(images):
    img.save(f"image_{i}_high_inference .png")

print("images saved")
