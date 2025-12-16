#!/usr/bin/env python3
"""
Train SDXL (or SDXL-Turbo) to generate images conditioned on EEG embeddings instead of text.

High-level:
EEG (B,C,H,W) -> EEGViT -> tokens/pooled -> Adapter -> (prompt_embeds, pooled_prompt_embeds)
Image -> VAE latents -> add noise -> UNet(noised_latents, t, encoder_hidden_states=prompt_embeds, added_cond_kwargs={...})
Loss: MSE(pred_noise, true_noise)

Notes:
- CPU training will be extremely slow.
- For SDXL-Turbo, guidance is typically 0 at inference; training still uses CFG-style conditioning tensors.
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


# ----------------------------
# 1) EEG Encoder (reuse your ViT style)
# ----------------------------

class EEGViT(nn.Module):
    """
    Minimal EEG encoder:
    Input:  (B, C, H, W) EEG in 2D grid form
    Output: tokens (B, S, D), pooled (B, D)
    """
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 8,
                 depth: int = 2, num_heads: int = 4):
        super().__init__()
        # Simple "patchify" conv; patch_size=1 means per-pixel tokenization
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads if num_heads else max(1, embed_dim // 64),  # override via config in real use
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, eeg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = eeg.shape[0]
        x = self.proj(eeg)              # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls.expand(B, -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls, x], dim=1)  # (B, 1+N, D)
        tokens = self.enc(tokens)
        tokens = self.norm(tokens)
        pooled = tokens[:, 0]  # CLS token
        return tokens, pooled


# ----------------------------
# 2) Adapter: EEG embedding -> SDXL conditioning tensors
# ----------------------------

class EEGToSDXLCondition(nn.Module):
    """
    Produces the tensors SDXL expects:
      - prompt_embeds: (B, S, cross_attention_dim)
      - pooled_prompt_embeds: (B, pooled_dim)
    """
    def __init__(self, eeg_dim: int, cross_attention_dim: int, pooled_dim: int):
        super().__init__()
        self.to_cross = nn.Sequential(
            nn.LayerNorm(eeg_dim),
            nn.Linear(eeg_dim, cross_attention_dim),
        )
        self.to_pooled = nn.Sequential(
            nn.LayerNorm(eeg_dim),
            nn.Linear(eeg_dim, pooled_dim),
        )

        # Optional learned "negative prompt" embeddings (can also be zeros)
        self.neg_token = nn.Parameter(torch.zeros(1, 1, cross_attention_dim))
        self.neg_pooled = nn.Parameter(torch.zeros(1, pooled_dim))

    def forward(self, tokens: torch.Tensor, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        prompt_embeds = self.to_cross(tokens)        # (B, S, cross_attention_dim)
        pooled_prompt_embeds = self.to_pooled(pooled)  # (B, pooled_dim)

        B, S, _ = prompt_embeds.shape
        negative_prompt_embeds = self.neg_token.expand(B, S, -1).contiguous()
        negative_pooled_prompt_embeds = self.neg_pooled.expand(B, -1).contiguous()

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }


# ----------------------------
# 3) Your dataset must yield (eeg, image)
# ----------------------------

class DummyEEGImageDataset(torch.utils.data.Dataset):
    """
    Replace this with your real paired dataset:
      eeg:   Float tensor (C,H,W)
      image: Float tensor (3, Himg, Wimg) in [-1, 1]
    """
    def __len__(self): return 128

    def __getitem__(self, idx: int):
        eeg = torch.randn(1, 32, 128)              # example EEG 2D layout
        image = torch.randn(3, 512, 512).clamp(-1, 1)
        return eeg, image


# -------------------------
# Scheduler helper
# -------------------------

def unwrap_scheduler(obj: Any) -> Any:
    """
    Diffusers version differences:
      - sometimes returns scheduler
      - sometimes returns (scheduler, dict)
      - sometimes returns (dict, scheduler) (rare stubs)
    This unwraps to the scheduler instance.
    """
    if isinstance(obj, tuple) and len(obj) == 2:
        a, b = obj
        if hasattr(a, "add_noise"):
            return a
        if hasattr(b, "add_noise"):
            return b
    return obj


# ----------------------------
# 4) Training
# ----------------------------

@dataclass
class TrainCfg:
    model_id: str = "stabilityai/sdxl-turbo"
    lr: float = 1e-4
    batch_size: int = 1
    epochs: int = 1
    grad_accum: int = 1
    num_workers: int = 0
    mixed_precision: bool = True
    save_path: str = "eeg_sdxl_cond.pt"


def main():
    cfg = TrainCfg()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = (device == "cuda") and cfg.mixed_precision
    dtype = torch.float16 if use_fp16 else torch.float32

    # Load SDXL pipeline, but we will train UNet forward directly.
    pipe = StableDiffusionXLPipeline.from_pretrained(cfg.model_id, dtype=dtype)  # or torch_dtype
    pipe.to(device)

    unet = pipe.unet
    vae = pipe.vae

    # Define for adapter init
    cross_attention_dim = int(unet.config.cross_attention_dim)
    pooled_dim = int(pipe.text_encoder_2.config.projection_dim)

    # Freeze VAE (common)
    vae.requires_grad_(False)
    vae.eval()

    # Scheduler for training objective
    raw = DDPMScheduler.from_config(pipe.scheduler.config)
    noise_scheduler = unwrap_scheduler(raw)

    # Robust way to get training timesteps across versions:
    if hasattr(noise_scheduler, "num_train_timesteps"):
        num_steps = int(noise_scheduler.num_train_timesteps)
    elif hasattr(noise_scheduler, "config") and hasattr(noise_scheduler.config, "num_train_timesteps"):
        num_steps = int(noise_scheduler.config.num_train_timesteps)
    else:
        # last resort: fall back to the pipeline scheduler's config
        num_steps = int(pipe.scheduler.config.num_train_timesteps)

        # Discover required conditioning dims from the pipeline
        cross_attention_dim = unet.config.cross_attention_dim
        pooled_dim = pipe.text_encoder_2.config.projection_dim

    # Build EEG encoder + adapter
    cpu_debug = (device == "cpu")

    eeg_dim = 128 if cpu_debug else 1024
    patch_size = 8 if cpu_debug else 1
    depth = 2 if cpu_debug else 12
    num_heads = 4 if cpu_debug else 16

    eeg_encoder = EEGViT(
        in_channels=1,
        embed_dim=eeg_dim,
        patch_size=patch_size,
        depth=depth,
        num_heads=num_heads,
    ).to(device, dtype=dtype)

    cond_adapter = EEGToSDXLCondition(
        eeg_dim=eeg_dim,
        cross_attention_dim=cross_attention_dim,
        pooled_dim=pooled_dim,
    ).to(device, dtype=dtype)

    # What to train?
    # Start by training ONLY eeg_encoder + adapter (freeze UNet) for stability.
    unet.requires_grad_(False)
    unet.eval()

    params = list(eeg_encoder.parameters()) + list(cond_adapter.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.lr)

    ds = DummyEEGImageDataset()
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    step = 0
    for epoch in range(cfg.epochs):
        for eeg, image in dl:
            eeg = eeg.to(device, dtype=dtype)
            image = image.to(device, dtype=dtype)

            with torch.no_grad():
                # VAE expects [0,1] images in many setups; SDXL VAE in diffusers typically uses [-1,1] inputs.
                # If your real images are [0,1], map to [-1,1] before this step.
                latents = vae.encode(image).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,),
                device=device, dtype=torch.long
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            tokens, pooled = eeg_encoder(eeg)
            cond = cond_adapter(tokens, pooled)

           # SDXL "time_ids" length is typically 6 (orig_h, orig_w, crop_y, crop_x, target_h, target_w)
            # Some variants may use 8; we'll infer it from the UNet config if possible, else default to 6.
            time_ids_dim = getattr(pipe.unet.config, "addition_time_embed_dim", None)

            # Most SDXL models use 6 values; treat this as the default metadata vector length.
            # The UNet internally projects it using addition_time_embed_dim; you just need the correct vector length.
            add_time_ids = torch.tensor(
                [image.shape[-2], image.shape[-1], 0, 0, image.shape[-2], image.shape[-1]],
                device=device,
                dtype=dtype,
            ).unsqueeze(0).repeat(bsz, 1)


            added_cond_kwargs = {
                "text_embeds": cond["pooled_prompt_embeds"],
                "time_ids": add_time_ids,
            }

            with torch.cuda.amp.autocast(enabled=use_fp16):
                # UNet forward: predict noise residual
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=cond["prompt_embeds"],
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            if step % 10 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.6f}")

            step += 1

    # Save your conditioning stack
    torch.save(
        {
            "eeg_encoder": eeg_encoder.state_dict(),
            "cond_adapter": cond_adapter.state_dict(),
            "cfg": cfg.__dict__,
            "cross_attention_dim": cross_attention_dim,
            "pooled_dim": pooled_dim,
            "eeg_dim": eeg_dim,
        },
        cfg.save_path,
    )
    print(f"Saved: {cfg.save_path}")


if __name__ == "__main__":
    main()
