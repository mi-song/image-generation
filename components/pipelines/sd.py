from typing import Callable, Optional

import torch

from components.encoders.sd import SDTextEncoder
from components.models.sd import SDModel
from components.samplers.base import SamplerBase


class SDPipeline:
    def __init__(self, model: SDModel, encoder: SDTextEncoder, sampler: SamplerBase):
        self.model = model
        self.encoder = encoder
        self.sampler = sampler

    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def run(
        self,
        null_prompt: str,
        prompt: str,
        cfg_guidance: float = 7.5,
        callback_fn: Optional[Callable] = None,
        src_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        emb = self.encoder.encode(null_prompt, prompt)

        src_latent = None
        if src_image is not None:
            with torch.no_grad():
                src_latent = self.model.encode(src_image.to(self.model.dtype).to(self.model.device))

        latent_shape = (1, 4, 64, 64)
        z = self.sampler.sample(
            uc=emb.uc,
            c=emb.c,
            cfg_guidance=cfg_guidance,
            latent_shape=latent_shape,
            model_kwargs=None,
            src_latent=src_latent,
            callback_fn=callback_fn,
        )

        with torch.no_grad():
            img = self.model.decode(z)
        return (img / 2 + 0.5).clamp(0, 1).detach().cpu()
