import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

small_model = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(small_model, torch_dtype=torch.bfloat16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")

prompts = [
    "Create an image of a futuristic city. The city has tall, shiny buildings made of glass and metal. The buildings have cool, modern shapes with curves and sharp angles. Some buildings have plants growing on them. The sky is dark blue, and flying cars move through the air, leaving light trails behind them. On the ground, people walk on wide, clean streets. They use high-tech gadgets and wear clothes that glow. Robots and drones help with jobs like deliveries and repairs.The city is full of bright neon lights in blue, pink, and green. These lights shine from signs, building edges, and paths, reflecting off the glass buildings. There are fast, quiet trains on elevated tracks for public transportation. There are also parks with unusual plants and trees, adding nature to the city.In the background, a huge, clear dome covers part of the city, maybe for climate control. The whole scene shows a place full of advanced technology and innovation, where tech is a natural part of daily life, making the city a center of progress and futuristic style.",
    "Create a picture of a mountain range at night. The scene features tall, rugged mountains with sharp peaks that reach up into the dark sky. The mountains are covered in snow, which glows faintly under the light of the full moon. The sky is clear and filled with countless stars, creating a stunning, starry backdrop. The Milky Way is visible, stretching across the sky like a luminous ribbon.At the base of the mountains, there is a dense forest of tall pine trees, their dark silhouettes contrasting with the snowy peaks. A calm, reflective lake lies in the foreground, mirroring the starry sky and the moonlit mountains. The water is still, creating a perfect reflection that enhances the serene atmosphere.In the distance, a small cabin with a softly glowing light in the window is nestled among the trees, adding a touch of warmth to the cold, tranquil night. The air is crisp and fresh, and the only sound is the gentle rustling of the trees and the occasional call of a night bird.The overall scene is peaceful and awe-inspiring, capturing the majestic beauty of the mountains under a starlit sky.",
    "A cyberpunk character with neon tattoos in a rain-soaked alley."
]

results = pipe(
    prompts,
    num_inference_steps=50,
    guidance_scale=3.5,
    height=512,
    width=512
)

images = results.images

# Save or display the images
for i, img in enumerate(images):
    img.save(f"image_{i}.png")  # Save each image
