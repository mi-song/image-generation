from typing import Any, Dict, Optional

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from .base import ModelBase, NoiseOutput


class SDModel(ModelBase):
    def __init__(
        self,
        model_key: str = "runwayml/stable-diffusion-v1-5",
        num_sampling: int = 50,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.device = device
        self.dtype = dtype

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=dtype).to(device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.prediction_type = getattr(self.unet.config, "prediction_type", "epsilon")

        sched = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self._total_alphas = sched.alphas_cumprod.clone()
        self._sigmas = (1 - self._total_alphas).sqrt() / self._total_alphas.sqrt()
        self._log_sigmas = self._sigmas.log()

        N_ts = len(sched.timesteps)
        sched.set_timesteps(num_sampling, device=device)
        self._skip = N_ts // num_sampling
        self._final_alpha_cumprod = sched.final_alpha_cumprod.to(device)
        sched.alphas_cumprod = torch.cat([torch.tensor([1.0]), sched.alphas_cumprod])
        self._scheduler = sched

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def total_alphas(self): return self._total_alphas
    @property
    def sigmas(self): return self._sigmas
    @property
    def log_sigmas(self): return self._log_sigmas
    @property
    def timesteps(self): return self._scheduler.timesteps
    @property
    def skip(self): return self._skip
    @property
    def scheduler(self): return self._scheduler

    def alpha(self, t) -> torch.Tensor:
        t_val = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        if t_val < 0:
            return self._final_alpha_cumprod
        return self._total_alphas[t_val].to(self.device)

    # ── UNet ─────────────────────────────────────────────────────────────────

    def _unet_forward(self, zt, t, uc, c):
        t_in = t.unsqueeze(0)
        if uc is None:
            out_c = self.unet(zt, t_in, encoder_hidden_states=c)["sample"]
            return out_c, out_c
        if c is None:
            out_uc = self.unet(zt, t_in, encoder_hidden_states=uc)["sample"]
            return out_uc, out_uc
        c_embed = torch.cat([uc, c], dim=0)
        z_in = torch.cat([zt] * 2)
        t_in = torch.cat([t_in] * 2)
        out = self.unet(z_in, t_in, encoder_hidden_states=c_embed)["sample"]
        return out.chunk(2)

    def predict_noise(self, zt, t, uc, c, model_kwargs=None) -> NoiseOutput:
        noise_uc, noise_c = self._unet_forward(zt, t, uc, c)
        if self.prediction_type == "v_prediction":
            at = self.alpha(t)
            noise_uc = at.sqrt() * noise_uc + (1 - at).sqrt() * zt
            noise_c  = at.sqrt() * noise_c  + (1 - at).sqrt() * zt
        return NoiseOutput(noise_uc, noise_c)

    def predict_raw(self, zt, t, uc, c, model_kwargs=None) -> NoiseOutput:
        out_uc, out_c = self._unet_forward(zt, t, uc, c)
        return NoiseOutput(out_uc, out_c)

    # ── VAE ──────────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x).latent_dist.sample() * 0.18215

    def decode(self, zt: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(zt / 0.18215).sample.float()
