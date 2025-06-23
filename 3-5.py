import os
import torch
import gc
from diffusers import StableDiffusion3Pipeline

# 🔧 Set PyTorch memory management env var (avoid memory fragmentation)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 🧹 Clear memory before starting (especially helpful in IDEs like PyCharm)
gc.collect()
torch.cuda.empty_cache()

# ✅ Use the smaller base model (better for 8GB GPU)
model_name = "stabilityai/stable-diffusion-3.5-large"

# 🛠 Load the pipeline with memory-efficient dtype
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16  # Lower memory usage
)

# ✅ Enable memory-saving features
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()  # Offload parts to CPU
# Only call this if xformers is installed
try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    print("⚠️ xformers not found. Skipping xformers-based attention optimization.")

# 🚀 Move model to GPU (memory offloading will auto-manage some CPU usage)
pipe = pipe.to("cpu")

# ✨ Prompt for the image
prompt = (
    "Create an image of a futuristic city. The city has tall, shiny buildings made of glass and metal. The buildings have cool, modern shapes with curves and sharp angles. Some buildings have plants growing on them. The sky is dark blue, and flying cars move through the air, leaving light trails behind them. On the ground, people walk on wide, clean streets. They use high-tech gadgets and wear clothes that glow. Robots and drones help with jobs like deliveries and repairs.The city is full of bright neon lights in blue, pink, and green. These lights shine from signs, building edges, and paths, reflecting off the glass buildings. There are fast, quiet trains on elevated tracks for public transportation. There are also parks with unusual plants and trees, adding nature to the city.In the background, a huge, clear dome covers part of the city, maybe for climate control. The whole scene shows a place full of advanced technology and innovation, where tech is a natural part of daily life, making the city a center of progress and futuristic style."
)

# 📐 Lower resolution for less VRAM usage (adjust if needed)
height = 1024
width = 1024

# 🎨 Generate image
results = pipe(
    prompt,
    num_inference_steps=100,
    guidance_scale=10,
    height=height,
    width=width
)

# 💾 Save the output image
image = results.images[0]
image.save("futuristic_city3.png")
print("✅ Image generated and saved as 'futuristic_city.png'")
