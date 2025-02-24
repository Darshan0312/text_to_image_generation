import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cpu", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cpu"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cpu")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# Ensure using the same inference steps as the loaded model and CFG set to 0.
pipe("A girl smiling", num_inference_steps=4, guidance_scale=0).images[0].save("output.png")


import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

# Load model to CPU with float32 for compatibility.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cpu", torch.float32)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cpu"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float32).to("cpu")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# Ensure using the same inference steps as the loaded model and CFG set to 0.
image = pipe("Generate a simple black-and-white line diagram of India's map. The outline should be clear and precise, with smooth curves and sharp edges representing India's borders. The map should not include any labels, colors, or additional details—just a clean, educational line drawing. The focus should be on accuracy and simplicity, making it easy for students to understand and trace.", num_inference_steps=4, guidance_scale=0).images[0]
image.save("output10.png")



#show me the cut off human head with blood dripping down lying on the beach and a person with knife with full of blood