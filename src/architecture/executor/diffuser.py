#!/usr/bin/env python3
"""
Generate one example image using a lightweight Stable Diffusion model from Hugging Face.

Model: stabilityai/stable-diffusion-2-1-base (lighter than SD 1.5/2.1 full variants)
Output: out.png
"""

import os
import torch
from typing import cast
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
# from diffusers.pipelines.stable_diffusion.pipeline_output import (
#     StableDiffusionPipelineOutput,
# )

MODEL_ID = "stabilityai/sdxl-turbo"  # relatively lightweight SD checkpoint
OUTFILE = "out.png"

def main():
    prompt = "An image of a car intersection from a street camera. Low resolution, mid day, light traffic."

    # Pick device + dtype
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    # If you use a gated model, set HF_TOKEN in your environment
    # export HF_TOKEN=...
    token = os.environ.get("HF_TOKEN", None)

    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)

    # pipe = StableDiffusionXLPipeline.from_pretrained(
    #     MODEL_ID,
    #     torch_dtype=dtype,
    #     use_safetensors=False,
    #     token=token,  # or remove this line if not needed
    # )

    # # Faster scheduler vs default
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # # Optional memory optimizations (safe to keep; may no-op on some setups)
    # try:
    #     pipe.enable_attention_slicing()
    # except Exception:
    #     pass

    pipe = pipe.to(device)

    # Generate
    result = cast(
        StableDiffusionPipelineOutput,
        pipe(
            prompt=prompt,
            num_inference_steps=4, # 25
            guidance_scale=0.0 # 7.5
            # width=512,
            # height=512,
        ),
    )

    image = result.images[0]

    image.save(OUTFILE)
    print(f"Saved: {OUTFILE} (device={device}, dtype={dtype})")

if __name__ == "__main__":
    main()
