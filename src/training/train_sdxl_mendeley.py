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
import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Callable, Optional, cast
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path

import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import hashlib
from peft import LoraConfig
from PIL import Image

IMG_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def debug_show_structure(p: str) -> None:
    obj = torch.load(p, map_location="cpu", weights_only=False)
    print("TOP KEYS:", obj.keys())
    for k in ["dataset", "images", "labels"]:
        v = obj.get(k, None)
        print(f"\n{k}: type={type(v)}")
        if isinstance(v, dict):
            print(f"  dict keys: {list(v.keys())[:50]}")
            # print one nested level
            for kk in list(v.keys())[:5]:
                vv = v[kk]
                if torch.is_tensor(vv):
                    print(f"  - {kk}: tensor shape={tuple(vv.shape)} dtype={vv.dtype}")
                elif isinstance(vv, (list, tuple)):
                    print(f"  - {kk}: {type(vv)} len={len(vv)} item0={type(vv[0]) if len(vv) else None}")
                else:
                    print(f"  - {kk}: type={type(vv)}")
        elif torch.is_tensor(v):
            print(f"  tensor shape={tuple(v.shape)} dtype={v.dtype}")
        elif isinstance(v, (list, tuple)):
            print(f"  {type(v)} len={len(v)} item0={type(v[0]) if len(v) else None}")



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

def pil_to_tensor_01(pil: Image.Image) -> torch.Tensor:
    arr = np.array(pil, dtype=np.float32) / 255.0   # (H,W,3) in [0,1]
    t = torch.from_numpy(arr).permute(2,0,1).contiguous()  # (3,H,W)
    return t


@dataclass(frozen=True)
class VEPSample:
    eeg_path: Path
    category: str      # e.g., "Apple", "Car", "Flower", "Human Face"
    stim_id: str       # e.g., "A1", "C2", "F1", "P2"


# ------------------ Mendeley VEP dataset ------------------

class MendeleyVEPDataset(torch.utils.data.Dataset):
    """
    Loader for:
      EEG Dataset for natural image recognition through Visual Stimuli (Mendeley)

    Expected extracted layout (as you showed):
      <mendeley_root>/
        VEP-DATA/VEP-DATA/
          Participant_info.xlsx
          VVIQuestionnaire.pdf
          stimuli_images/
            Apple/A1.png
            Car/C1.jpg
            Flower/F1.jpg
            Human Face/P1.png
            ...
          VEP-CSV/<Category>/<StimulusID>/sub##_X#.csv (+ .json)
          VEP-EDF/<Category>/<StimulusID>/sub##_X#.edf (+ .json)

    Notes:
    - This dataset DOES include your own stimuli_images folder (you added it). If stimuli_dir
      is not provided, it defaults to: <mendeley_root>/VEP-DATA/VEP-DATA/stimuli_images
    - EEG returned as torch.float32 with shape (1, C, T).
    - Image returned as torch.float32 with shape (3, H, W) in [-1, 1].
    """

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    def __init__(
        self,
        mendeley_root: str | Path,
        stimuli_dir: Optional[str | Path] = None,
        *,
        fmt: str = "csv",  # "csv" or "edf"
        image_size: int = 512,
        prefer_csv: bool = True,
        strict_images: bool = True,  # if False, allows A2->A1, C2->C1, F2->F1, P2->P1 fallback
    ):
        self.root = Path(mendeley_root).expanduser().resolve()
        self.fmt = fmt.lower().strip()
        self.image_size = int(image_size)
        self.strict_images = bool(strict_images)

        if not self.root.exists():
            raise FileNotFoundError(f"Mendeley root not found: {self.root}")

        base = self._find_vep_data_base(self.root)
        if base is None:
            raise FileNotFoundError(
                "Could not locate 'VEP-CSV' / 'VEP-EDF' under the provided root. "
                "Expected something like: <root>/VEP-DATA/VEP-DATA/VEP-CSV/..."
            )

        # Default stimuli_dir to the in-dataset stimuli_images you created
        if stimuli_dir is None:
            self.stimuli_dir = base / "stimuli_images"
        else:
            self.stimuli_dir = Path(stimuli_dir).expanduser().resolve()

        if not self.stimuli_dir.exists():
            raise FileNotFoundError(
                f"stimuli_dir not found: {self.stimuli_dir}\n"
                f"(If you put stimuli_images inside the dataset, you can omit --stimuli_dir "
                f"and it will default to: {base / 'stimuli_images'})"
            )

        vep_csv = base / "VEP-CSV"
        vep_edf = base / "VEP-EDF"

        if self.fmt not in ("csv", "edf"):
            raise ValueError(f"fmt must be 'csv' or 'edf', got {self.fmt}")

        if self.fmt == "csv":
            if not vep_csv.exists():
                # fallback if user picked csv but only edf exists
                if vep_edf.exists() and prefer_csv is False:
                    self.fmt = "edf"
                else:
                    raise FileNotFoundError(f"CSV folder not found: {vep_csv}")
            self.data_root = vep_csv
        else:
            if not vep_edf.exists():
                raise FileNotFoundError(f"EDF folder not found: {vep_edf}")
            self.data_root = vep_edf

        self.samples: list[dict[str, Any]] = self._index_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No EEG files found under: {self.data_root}")

        # Image preprocessing: force to image_size x image_size; output [-1, 1]
        self._img_tf: Callable[[Image.Image], torch.Tensor] = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    @staticmethod
    def _find_vep_data_base(root: Path) -> Path | None:
        """
        Returns the directory that contains VEP-CSV and/or VEP-EDF.
        In your structure: <root>/VEP-DATA/VEP-DATA/
        """
        cand = root / "VEP-DATA" / "VEP-DATA"
        if cand.exists() and cand.is_dir():
            if (cand / "VEP-CSV").exists() or (cand / "VEP-EDF").exists():
                return cand

        # Otherwise, search
        for p in root.rglob("VEP-CSV"):
            if p.is_dir():
                return p.parent
        for p in root.rglob("VEP-EDF"):
            if p.is_dir():
                return p.parent
        return None

    def _index_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        eeg_paths = sorted(self.data_root.rglob("*.csv" if self.fmt == "csv" else "*.edf"))

        for eeg_path in eeg_paths:
            # .../<Category>/<StimulusID>/sub##_A1.csv
            stim_id = eeg_path.parent.name           # A1, A2, C1, C2, F1, F2, P1, P2
            category = eeg_path.parent.parent.name   # Apple, Car, Flower, Human Face

            json_path = eeg_path.with_suffix(".json")
            samples.append({
                "eeg_path": eeg_path,
                "json_path": json_path if json_path.exists() else None,
                "category": category,
                "stim_id": stim_id,
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        eeg_path: Path = s["eeg_path"]
        category: str = s["category"]
        stim_id: str = s["stim_id"]

        eeg = self._load_eeg(eeg_path)
        eeg = self._ensure_eeg_ct(eeg)
        eeg_t = eeg.unsqueeze(0)  # (1, C, T)

        img_path = self._resolve_stimulus_image(category, stim_id, eeg_path)
        img_t = self._load_and_preprocess_image(img_path)

        return eeg_t, img_t

    # ------------------ EEG loading ------------------

    def _load_eeg(self, path: Path) -> torch.Tensor:
        if self.fmt == "csv":
            return self._load_csv_eeg(path)
        return self._load_edf_eeg(path)

    def _load_csv_eeg(self, path: Path) -> torch.Tensor:
        # Try detect header (non-numeric first token)
        first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:2]
        skip = 0
        if first:
            tok = first[0].split(",")[0].strip()
            try:
                float(tok)
            except Exception:
                skip = 1

        try:
            arr = np.loadtxt(path, delimiter=",", skiprows=skip, dtype=np.float32)
        except Exception:
            arr = np.loadtxt(path, delimiter=";", skiprows=skip, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr[None, :]

        # Heuristic: if time is rows and channels is cols, flip to (C, T)
        if arr.shape[0] > arr.shape[1] and arr.shape[1] <= 256:
            arr = arr.T

        return torch.tensor(arr, dtype=torch.float32)

    def _load_edf_eeg(self, path: Path) -> torch.Tensor:
        try:
            import mne  # type: ignore
        except Exception as e:
            raise ImportError("EDF reading requires 'mne'. Install with: uv pip install mne") from e

        raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")

        # Force numpy array (and make type-checkers happy)
        data = np.asarray(raw.get_data(), dtype=np.float32)  # (C, T)

        return torch.from_numpy(data)  # already float32


    @staticmethod
    def _ensure_eeg_ct(eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim == 1:
            return eeg.unsqueeze(0)
        if eeg.ndim == 2:
            return eeg
        return eeg.reshape(eeg.shape[0], -1)

    # ------------------ image resolution/loading ------------------

    def _resolve_stimulus_image(self, category: str, stim_id: str, eeg_path: Path) -> Path:
        """
        Your current stimuli layout is:
          stimuli_images/<Category>/<StimulusID>.<ext>
        (not a folder per StimulusID)
        """

        # 1) stimuli_dir/<Category>/<StimulusID>.<ext>
        cat_dir = self.stimuli_dir / category
        for ext in self.IMG_EXTS:
            cand = cat_dir / f"{stim_id}{ext}"
            if cand.exists():
                return cand

        # 2) stimuli_dir/<StimulusID>.<ext> (optional alternate)
        for ext in self.IMG_EXTS:
            cand = self.stimuli_dir / f"{stim_id}{ext}"
            if cand.exists():
                return cand

        # 3) stimuli_dir/<Category>.<ext> (category-level fallback)
        for ext in self.IMG_EXTS:
            cand = self.stimuli_dir / f"{category}{ext}"
            if cand.exists():
                return cand

        # 4) Optional fallback: A2->A1, C2->C1, F2->F1, P2->P1
        if not self.strict_images:
            fallback = self._fallback_stim_id(stim_id)
            if fallback != stim_id:
                return self._resolve_stimulus_image(category, fallback, eeg_path)

        raise FileNotFoundError(
            f"Could not find a stimulus image for category='{category}', stim_id='{stim_id}'.\n"
            f"stimuli_dir={self.stimuli_dir}\n"
            f"Tried:\n"
            f"  - {cat_dir}/{stim_id}.*\n"
            f"  - {self.stimuli_dir}/{stim_id}.*\n"
            f"  - {self.stimuli_dir}/{category}.*\n"
            f"Extensions: {self.IMG_EXTS}\n"
            f"If you only have A1/C1/F1/P1 images, set strict_images=False."
        )

    @staticmethod
    def _fallback_stim_id(stim_id: str) -> str:
        if len(stim_id) == 2 and stim_id[1] == "2":
            return stim_id[0] + "1"
        return stim_id

    def _load_and_preprocess_image(self, img_path: Path) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB")
        return self._img_tf(img)



def make_splits(dataset: MendeleyVEPDataset, test_size: float, seed: int) -> tuple[Subset, Subset]:
    """
    Deterministically split dataset into train/test subsets.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be in (0,1), got {test_size}")

    n = len(dataset)
    n_test = max(1, int(round(n * test_size)))
    n_train = n - n_test
    if n_train < 1:
        raise ValueError(f"Not enough data for train split: n={n}, test_size={test_size}")

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    return Subset(dataset, train_idx), Subset(dataset, test_idx)


# ---------------------------
# Visualization
# ---------------------------


def get_eeg_img(ds, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    sample = ds[idx]
    eeg, img = cast(tuple[torch.Tensor, torch.Tensor], sample)
    return eeg, img


def tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
    """
    img_t: (3,H,W) in [-1, 1] float
    """
    img = img_t.detach().cpu()
    img = (img + 1.0) * 0.5  # [0,1]
    img = img.clamp(0, 1)
    img = (img * 255.0).to(torch.uint8)
    img = img.permute(1, 2, 0).contiguous().numpy()  # (H,W,3)
    return Image.fromarray(img)


@torch.no_grad()
def generate_images_from_eeg(
    pipe: StableDiffusionXLPipeline,
    eeg: torch.Tensor,                 # (B,1,C,T) per your dataset
    eeg_encoder: nn.Module,
    cond_adapter: nn.Module,
    device: str,
    dtype: torch.dtype,
    cond_scale: float = 1.0,
    num_inference_steps: int = 4,
    guidance_scale: float = 0.0,       # SDXL-Turbo often uses 0.0; SDXL can use ~5-7
    seed: int = 0,
) -> list[Image.Image]:
    """
    Returns list of PIL images, one per batch item.
    Uses SDXL prompt_embeds + pooled_prompt_embeds directly (no text prompts).
    """
    pipe.unet.eval()
    eeg_encoder.eval()
    cond_adapter.eval()

    eeg = eeg.to(device, dtype=dtype)

    tokens, pooled = eeg_encoder(eeg)
    cond = cond_adapter(tokens, pooled)

    if cond_scale != 1.0:
        cond["prompt_embeds"] = cond["prompt_embeds"] * cond_scale
        cond["pooled_prompt_embeds"] = cond["pooled_prompt_embeds"] * cond_scale

    # Important: pass BOTH positive and negative embeds for SDXL
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    out = pipe(
        prompt_embeds=cond["prompt_embeds"],
        pooled_prompt_embeds=cond["pooled_prompt_embeds"],
        negative_prompt_embeds=cond["negative_prompt_embeds"],
        negative_pooled_prompt_embeds=cond["negative_pooled_prompt_embeds"],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pil",
        return_dict=True,   # <-- key
    )

    imgs = out.images if hasattr(out, "images") else out[0] # type: ignore
    return list(imgs) # type: ignore


def run_inference_and_visualize(
    pipe: StableDiffusionXLPipeline,
    eeg: torch.Tensor,                 # (B,1,C,T)
    gt_img: torch.Tensor,              # (B,3,H,W) in [-1,1]
    eeg_encoder: nn.Module,
    cond_adapter: nn.Module,
    device: str,
    dtype: torch.dtype,
    out_path: str,
    cond_scale: float = 1.0,
    num_inference_steps: int = 4,
    guidance_scale: float = 0.0,
    seed: int = 0,
    title: str | None = None,
) -> None:
    """
    Saves a side-by-side PNG: Ground Truth (answer) vs Model Output.
    Uses the first item in the batch.
    """
    preds = generate_images_from_eeg(
        pipe=pipe,
        eeg=eeg,
        eeg_encoder=eeg_encoder,
        cond_adapter=cond_adapter,
        device=device,
        dtype=dtype,
        cond_scale=cond_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    pred_pil = preds[0]
    gt_pil = tensor_to_pil(gt_img[0])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=(10, 5))
    if title:
        fig.suptitle(title)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(gt_pil)
    ax1.set_title("Ground Truth (Answer)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(pred_pil)
    ax2.set_title("Model Output")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[inference] Saved visualization: {out_path}")


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
# Config
# ----------------------------

@dataclass
class DimCfg:
    """
    Central place to control dimensionality + depth for EEG encoder + conditioning.

    Notes:
    - SDXL requires prompt_embeds last-dim == unet.config.cross_attention_dim (usually 2048)
    - SDXL requires pooled_prompt_embeds last-dim == text_encoder_2.projection_dim (usually 1280)
    - The only dims you *really* control here are EEGViT dims and token count (patch_size).
    """

    # Toggle profiles
    cpu_debug: bool = True  # auto-set in main based on device if you want

    # EEG input
    eeg_in_channels: int = 1

    # EEGViT "debug" profile (CPU)
    eeg_dim_cpu: int = 128
    eeg_depth_cpu: int = 2
    eeg_heads_cpu: int = 4
    eeg_patch_cpu: int = 16   # larger patch => fewer tokens => less attention cost

    # EEGViT "full" profile (GPU)
    eeg_dim_gpu: int = 4096
    eeg_depth_gpu: int = 6
    eeg_heads_gpu: int = 16
    eeg_patch_gpu: int = 8    # smaller patch => more tokens => more capacity

    # Conditioning strength (scale down conditioning vectors if needed)
    cond_scale: float = 1.0

    def resolve_eeg(self, device: str) -> tuple[int, int, int, int]:
        """Return (embed_dim, depth, heads, patch_size) based on device/profile."""
        use_cpu = (device == "cpu") if self.cpu_debug else False
        if use_cpu:
            return (self.eeg_dim_cpu, self.eeg_depth_cpu, self.eeg_heads_cpu, self.eeg_patch_cpu)
        return (self.eeg_dim_gpu, self.eeg_depth_gpu, self.eeg_heads_gpu, self.eeg_patch_gpu)

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

    # LoRA rank schedule: increase over time
    lora_ranks: tuple[int, ...] = (4, 8, 16)
    lora_stage_epochs: tuple[int, ...] = (1, 1, 1)
    lora_alpha: int = 8


# ----------------------------
# 4) Training
# ----------------------------

def enable_unet_lora_peft(unet, rank: int, alpha: int):
    """
    Uses diffusers' PEFT integration to add LoRA to UNet attention layers.
    This is the most version-robust way to do LoRA for SDXL.
    """
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    print("UNet trainable:", sum(p.numel() for p in unet.parameters() if p.requires_grad))

    unet.set_adapters(["default"])  # ensures adapter active
    return unet


def inspect_lora_grads(unet, max_items=5):
    items = []
    for name, p in unet.named_parameters():
        if "lora_A" in name and p.requires_grad:
            if p.grad is None:
                items.append((name, "grad=None"))
            else:
                items.append((name, float(p.grad.abs().sum())))
        if len(items) >= max_items:
            break
    return items


@torch.no_grad()
def eval_one_epoch(
    test_dl: DataLoader,
    device: str,
    dtype: torch.dtype,
    vae,
    unet,
    noise_scheduler,
    num_steps: int,
    eeg_encoder,
    cond_adapter,
    cond_scale: float,
):
    unet.eval()
    eeg_encoder.eval()
    cond_adapter.eval()

    losses = []
    for eeg, image in test_dl:
        eeg = eeg.to(device, dtype=dtype)
        image = image.to(device, dtype=dtype)

        bsz = image.shape[0]

        latents = vae.encode(image).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, num_steps, (bsz,), device=device, dtype=torch.long)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        tokens, pooled = eeg_encoder(eeg)
        cond = cond_adapter(tokens, pooled)

        if cond_scale != 1.0:
            cond["prompt_embeds"] = cond["prompt_embeds"] * cond_scale
            cond["pooled_prompt_embeds"] = cond["pooled_prompt_embeds"] * cond_scale

        add_time_ids = torch.tensor(
            [image.shape[-2], image.shape[-1], 0, 0, image.shape[-2], image.shape[-1]],
            device=device,
            dtype=dtype,
        ).unsqueeze(0).repeat(bsz, 1)

        added_cond_kwargs = {
            "text_embeds": cond["pooled_prompt_embeds"],
            "time_ids": add_time_ids,
        }

        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=cond["prompt_embeds"],
            added_cond_kwargs=added_cond_kwargs,
        ).sample

        loss = F.mse_loss(noise_pred.float(), noise.float())
        losses.append(float(loss))

    return sum(losses) / max(1, len(losses))


def main():
    cfg = TrainCfg()
    dims = DimCfg()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke_test", 
        action="store_true", 
        help="Run 1 batch on CPU for sanity check"
    )
    parser.add_argument(
        "--data_npz",
        type=str,
        default="preprocessed_data/brain_image_dataset.npz",
        help="Path to preprocessed .npz dataset with eeg/images/labels.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Dataloader workers (keep 0 on Windows if you hit worker issues).",
    )
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2, 
        help="Fraction for test split (0-1)."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=83, 
        help="Random seed for deterministic split."
    )
    parser.add_argument(
        "--infer_after",
        action="store_true",
        help="Run inference + visualization (SMOKE_TEST: on the smoke batch; training: on a test sample) and save a PNG.",
    )
    parser.add_argument(
        "--infer_idx",
        type=int,
        default=0,
        help="Index into test_ds to visualize after training (ignored in SMOKE_TEST).",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=4,
        help="Diffusion inference steps (Turbo often 1-4; SDXL often 20-50).",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=0.0,
        help="CFG guidance scale (Turbo often 0.0; SDXL often ~5-7).",
    )
    parser.add_argument(
        "--infer_outdir",
        type=str,
        default="inference_outputs",
        help="Directory to write inference visualizations.",
    )
    parser.add_argument(
        "--eeg_imagenet_parts", 
        nargs="+", 
        default=[
            "datasets/eeg_imagenet/EEG-ImageNet_1.pth",
        ], 
        help="One or more EEG-ImageNet part .pth files (1 and/or 2)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mendeley",
        choices=["mendeley", "eeg_imagenet", "npz"],
        help="Which dataset loader to use."
    )
    parser.add_argument(
        "--mendeley_root",
        type=str,
        default="datasets/EEG Dataset for natural image recognition through Visual Stimuli",
        help="Root folder where the Mendeley dataset was extracted."
    )
    parser.add_argument(
        "--mendeley_format",
        type=str,
        default="csv",
        choices=["csv", "edf"],
        help="Whether to load EEG from VEP-CSV or VEP-EDF."
    )
    parser.add_argument(
        "--stimuli_dir",
        type=str,
        default="./datasets/EEG Dataset for natural image recognition through Visual Stimuli/VEP-DATA/VEP-DATA/stimuli_images",
        help="Folder containing stimulus images (see script docstring for expected layout)."
    )
    parser.add_argument(
        "--imagenet_root",
        type=str,
        default=None,
        help="Path to your local ImageNet root folder (required if --dataset eeg_imagenet)."
    )


    SMOKE_TEST = True  # set False for real training
    try:
        args = parser.parse_args()
    except SystemExit:
        sys.exit(1)

    SMOKE_TEST = args.smoke_test 
    cfg.num_workers = args.num_workers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if SMOKE_TEST:
        device = "cpu"
    use_fp16 = (device == "cuda") and cfg.mixed_precision
    dtype = torch.float16 if use_fp16 else torch.float32

    # Load SDXL pipeline; we'll call UNet directly
    pipe = StableDiffusionXLPipeline.from_pretrained(cfg.model_id, dtype=dtype)
    pipe.to(device)

    unet = pipe.unet
    vae = pipe.vae

    # Conditioning dims
    cross_attention_dim = int(unet.config.cross_attention_dim)
    pooled_dim = int(pipe.text_encoder_2.config.projection_dim)

    # Freeze VAE
    vae.requires_grad_(False)
    vae.eval()

    # Noise scheduler (robust unwrap across diffusers versions)
    raw = DDPMScheduler.from_config(pipe.scheduler.config)
    noise_scheduler = unwrap_scheduler(raw)

    # Training timesteps count
    if hasattr(noise_scheduler, "config") and hasattr(noise_scheduler.config, "num_train_timesteps"):
        num_steps = int(noise_scheduler.config.num_train_timesteps)
    elif hasattr(noise_scheduler, "num_train_timesteps"):
        num_steps = int(noise_scheduler.num_train_timesteps)
    else:
        num_steps = int(pipe.scheduler.config.num_train_timesteps)

    eeg_dim, depth, num_heads, patch_size = dims.resolve_eeg(device)

    eeg_encoder = EEGViT(
        in_channels=dims.eeg_in_channels,
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

    # Freeze base UNet weights (LoRA will re-enable only adapter params)
    unet.requires_grad_(False)

    # Only attach LoRA early if we're doing SMOKE_TEST.
    # For real training, LoRA is attached per-stage below.
    if SMOKE_TEST:
        unet = enable_unet_lora_peft(unet, rank=cfg.lora_ranks[0], alpha=cfg.lora_alpha)
        unet.train()

        # Ensure only LoRA params trainable
        for n, p in unet.named_parameters():
            p.requires_grad = ("lora_" in n)

        lora_params = [p for n, p in unet.named_parameters() if p.requires_grad]
        params = list(eeg_encoder.parameters()) + list(cond_adapter.parameters()) + lora_params
        optim = torch.optim.AdamW(params, lr=cfg.lr)
    else:
        # We'll build the optimizer inside each LoRA stage, after attaching the adapter.
        optim = None

    # debug_show_structure(args.eeg_imagenet_parts[0])
    # sys.exit(0)

    # ------------------ dataset selection ------------------
    ds = MendeleyVEPDataset(
        mendeley_root=args.mendeley_root,
        stimuli_dir=None,
        fmt=args.mendeley_format,
        image_size=512,
        strict_images=False
    )

    e0, im0 = ds[0]
    print("Example EEG:", e0.shape, e0.dtype, "Example IMG:", im0.shape, im0.dtype, im0.min().item(), im0.max().item())
    train_ds, test_ds = make_splits(ds, test_size=args.test_size, seed=args.seed)

    print(f"Loaded dataset: {len(ds)} examples")
    print(f"Train: {len(train_ds)}  Test: {len(test_ds)}")

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f"Loaded dataset: {len(ds)} examples")

    # AMP + scaler only on CUDA
    if use_fp16:
        scaler = torch.amp.GradScaler("cuda", enabled=True) # type: ignore
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) # type: ignore
    else:
        scaler = None
        # dummy context manager
        from contextlib import nullcontext
        autocast_ctx = nullcontext()

    # SMOKE_TEST: fetch exactly one batch once
    if SMOKE_TEST:
        eeg, image = next(iter(train_dl))
        eeg = eeg.to(device, dtype=dtype)
        image = image.to(device, dtype=dtype)

        print("=== SMOKE TEST BATCH ===")
        print("EEG:", eeg.shape, eeg.dtype, eeg.device)
        print("IMG:", image.shape, image.dtype, image.device)

        # One step
        bsz = image.shape[0]

        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, num_steps, (bsz,), device=device, dtype=torch.long)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        tokens, pooled = eeg_encoder(eeg)
        cond = cond_adapter(tokens, pooled)

        if dims.cond_scale != 1.0:
            cond["prompt_embeds"] = cond["prompt_embeds"] * dims.cond_scale
            cond["pooled_prompt_embeds"] = cond["pooled_prompt_embeds"] * dims.cond_scale

        print("tokens:", tokens.shape)
        print("pooled:", pooled.shape)
        print("prompt_embeds:", cond["prompt_embeds"].shape)
        print("pooled_prompt_embeds:", cond["pooled_prompt_embeds"].shape)

        # SDXL time_ids workaround (6-dim metadata vector)
        add_time_ids = torch.tensor(
            [image.shape[-2], image.shape[-1], 0, 0, image.shape[-2], image.shape[-1]],
            device=device,
            dtype=dtype,
        ).unsqueeze(0).repeat(bsz, 1)

        added_cond_kwargs = {
            "text_embeds": cond["pooled_prompt_embeds"],
            "time_ids": add_time_ids,
        }

        assert optim is not None, "Optimizer must be initialized in SMOKE_TEST"
        optim.zero_grad(set_to_none=True)

        with autocast_ctx:
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=cond["prompt_embeds"],
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            loss = F.mse_loss(noise_pred.float(), noise.float())

        if scaler is not None:
            scaler.scale(loss).backward()
            for n,p in unet.named_parameters():
                if "lora_B" in n and p.requires_grad:
                    print(n, p.grad is None, float(p.grad.abs().sum()) if p.grad is not None else None)
                    break
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            for n,p in unet.named_parameters():
                if "lora_B" in n and p.requires_grad:
                    print(n, p.grad is None, float(p.grad.abs().sum()) if p.grad is not None else None)
                    break
            optim.step()

        print("loss:", float(loss.detach()))
        print("✅ SMOKE TEST COMPLETE (one forward/backward/step)")

        if args.infer_after:
            out_png = os.path.join(args.infer_outdir, "smoke_inference.png")
            run_inference_and_visualize(
                pipe=pipe,
                eeg=eeg,
                gt_img=image,
                eeg_encoder=eeg_encoder,
                cond_adapter=cond_adapter,
                device=device,
                dtype=dtype,
                out_path=out_png,
                cond_scale=dims.cond_scale,
                num_inference_steps=args.infer_steps,
                guidance_scale=args.guidance,
                seed=args.seed,
                title="SMOKE_TEST: GT vs EEG->SDXL Output",
            )


        lora_state = {k: v.cpu() for k, v in unet.state_dict().items() if "lora" in k.lower()}

        # Save weights even in smoke test (handy)
        torch.save(
            {
                "eeg_encoder": eeg_encoder.state_dict(),
                "cond_adapter": cond_adapter.state_dict(),
                "unet_lora_state": lora_state,
                "cfg": cfg.__dict__,
                "cross_attention_dim": cross_attention_dim,
                "pooled_dim": pooled_dim,
                "eeg_dim": eeg_dim,
                "patch_size": patch_size,
                "depth": depth,
                "num_heads": num_heads,
            },
            cfg.save_path,
        )
        print(f"Saved: {cfg.save_path}")
        return

    # ============================
    # REAL TRAINING (LoRA stages)
    # ============================

    assert len(cfg.lora_ranks) == len(cfg.lora_stage_epochs), \
        "lora_ranks and lora_stage_epochs must have same length"

    global_step = 0

    for stage_i, (rank, stage_epochs) in enumerate(zip(cfg.lora_ranks, cfg.lora_stage_epochs)):
        print(f"\n== LoRA stage {stage_i+1}/{len(cfg.lora_ranks)}: rank={rank}, epochs={stage_epochs} ==")

        # Remove any prior adapters (start fresh) - safest across versions
        if hasattr(unet, "delete_adapters"):
            unet.delete_adapters()

        # Add LoRA to UNet (base UNet still frozen)
        unet = enable_unet_lora_peft(unet, rank=rank, alpha=cfg.lora_alpha)
        unet.train()

        # Ensure only LoRA params trainable
        for n, p in unet.named_parameters():
            p.requires_grad = ("lora_" in n)

        # Rebuild optimizer because trainable params changed
        lora_params = [
            p for n, p in unet.named_parameters() # type: ignore
            if p.requires_grad and ("lora" in n.lower() or "adapter" in n.lower())
        ]
        print("LoRA trainable params:", sum(p.numel() for p in lora_params))
        assert len(lora_params) > 0, "No trainable LoRA params found—LoRA was not attached correctly."
        params = list(eeg_encoder.parameters()) + list(cond_adapter.parameters()) + lora_params
        optim = torch.optim.AdamW(params, lr=cfg.lr)

        for epoch in range(stage_epochs):
            eeg_encoder.train()
            cond_adapter.train()

            for eeg, image in tqdm(train_dl, desc=f"Stage {stage_i+1} Epoch {epoch+1}/{stage_epochs}"):
                eeg = eeg.to(device, dtype=dtype)
                image = image.to(device, dtype=dtype)
                bsz = image.shape[0]

                with torch.no_grad():
                    latents = vae.encode(image).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, num_steps, (bsz,), device=device, dtype=torch.long)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                tokens, pooled = eeg_encoder(eeg)
                cond = cond_adapter(tokens, pooled)

                if dims.cond_scale != 1.0:
                    cond["prompt_embeds"] = cond["prompt_embeds"] * dims.cond_scale
                    cond["pooled_prompt_embeds"] = cond["pooled_prompt_embeds"] * dims.cond_scale

                add_time_ids = torch.tensor(
                    [image.shape[-2], image.shape[-1], 0, 0, image.shape[-2], image.shape[-1]],
                    device=device,
                    dtype=dtype,
                ).unsqueeze(0).repeat(bsz, 1)

                added_cond_kwargs = {
                    "text_embeds": cond["pooled_prompt_embeds"],
                    "time_ids": add_time_ids,
                }

                optim.zero_grad(set_to_none=True)

                with autocast_ctx:
                    noise_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=cond["prompt_embeds"],
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                    loss = F.mse_loss(noise_pred.float(), noise.float())

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()

                if global_step % 10 == 0:
                    print(f"stage={stage_i} rank={rank} step={global_step} loss={loss.item():.6f}")
                global_step += 1

            # Optional: evaluate after each stage-epoch
            test_loss = eval_one_epoch(
                test_dl=test_dl,
                device=device,
                dtype=dtype,
                vae=vae,
                unet=unet,
                noise_scheduler=noise_scheduler,
                num_steps=num_steps,
                eeg_encoder=eeg_encoder,
                cond_adapter=cond_adapter,
                cond_scale=dims.cond_scale,
            )
            print(f"[stage {stage_i+1} rank={rank} epoch {epoch+1}] test_loss={test_loss:.6f}")

    # Save final
    torch.save(
        {
            "eeg_encoder": eeg_encoder.state_dict(),
            "cond_adapter": cond_adapter.state_dict(),
            "cfg": cfg.__dict__,
            "cross_attention_dim": cross_attention_dim,
            "pooled_dim": pooled_dim,
            "eeg_dim": eeg_dim,
            "patch_size": patch_size,
            "depth": depth,
            "num_heads": num_heads,
        },
        cfg.save_path,
    )
    print(f"Saved: {cfg.save_path}")

    if args.infer_after:
        # pick a deterministic test sample
        idx = int(max(0, min(args.infer_idx, len(test_ds) - 1)))
        eeg_one, img_one = get_eeg_img(test_ds, idx)  # each is unbatched
        eeg_b = eeg_one.unsqueeze(0).to(device, dtype=dtype)
        img_b = img_one.unsqueeze(0).to(device, dtype=dtype)

        out_png = os.path.join(args.infer_outdir, f"trained_inference_idx{idx}.png")
        run_inference_and_visualize(
            pipe=pipe,
            eeg=eeg_b,
            gt_img=img_b,
            eeg_encoder=eeg_encoder,
            cond_adapter=cond_adapter,
            device=device,
            dtype=dtype,
            out_path=out_png,
            cond_scale=dims.cond_scale,
            num_inference_steps=args.infer_steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            title=f"TRAINED: GT vs EEG->SDXL Output (test idx={idx})",
        )




if __name__ == "__main__":
    main()
