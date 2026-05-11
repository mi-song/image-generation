from typing import Callable, Optional, Tuple

import torch

from components.encoders.base import TextEmbedding
from components.encoders.sdxl import SDXLTextEncoder
from components.models.sdxl import SDXLModel
from components.samplers.base import SamplerBase


class SDXLPipeline:
    def __init__(self, model: SDXLModel, encoder: SDXLTextEncoder, sampler: SamplerBase):
        self.model = model
        self.encoder = encoder
        self.sampler = sampler

    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def run(
        self,
        null_prompt: str,
        prompt: str,
        cfg_guidance: float = 7.5,
        target_size: Optional[Tuple[int, int]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        clip_skip: Optional[int] = None,
        src_image: Optional[torch.Tensor] = None,
        callback_fn: Optional[Callable] = None,
        # Hires fix
        hires_fix: bool = False,
        hires_scale: float = 0.75,
        hires_strength: float = 0.5,
    ) -> torch.Tensor:
        """Returns an image tensor in [0, 1] on CPU."""

        H = self.model.default_sample_size * self.model.vae_scale_factor
        W = H
        target_size   = target_size   or (H, W)
        original_size = original_size or target_size

        # 1. Text encoding
        emb: TextEmbedding = self.encoder.encode(null_prompt, prompt, clip_skip=clip_skip)

        # 2. SDXL conditioning kwargs (always built for final target_size)
        model_kwargs = self._build_model_kwargs(
            emb, cfg_guidance, original_size, target_size, crops_coords_top_left,
            negative_original_size, negative_crops_coords_top_left,
        )

        # 3. Source latent for inversion-based samplers
        src_latent = None
        if src_image is not None:
            with torch.no_grad():
                src_latent = self.model.encode(
                    src_image.to(self.model.dtype).to(self.model.device)
                )

        # 4. First pass latent shape (scaled down if hires fix)
        if hires_fix:
            first_h = int(target_size[0] * hires_scale // 8) * 8
            first_w = int(target_size[1] * hires_scale // 8) * 8
        else:
            first_h, first_w = target_size
        h_lat = first_h // self.model.vae_scale_factor
        w_lat = first_w // self.model.vae_scale_factor
        latent_shape = (1, 4, h_lat, w_lat)

        # 5. First pass sampling
        z = self.sampler.sample(
            uc=emb.uc,
            c=emb.c,
            cfg_guidance=cfg_guidance,
            latent_shape=latent_shape,
            model_kwargs=model_kwargs,
            src_latent=src_latent,
            callback_fn=callback_fn,
        )

        if not hires_fix:
            with torch.no_grad():
                img = self.model.decode(z)
            return (img / 2 + 0.5).clamp(0, 1).detach().cpu()

        # 6. Hires fix: decode → upscale → re-encode → img2img
        with torch.no_grad():
            img_first = self.model.decode(z).float()  # (1,C,H,W) in [-1, 1]

        img_up = torch.nn.functional.interpolate(
            img_first, size=(target_size[0], target_size[1]),
            mode="bicubic", align_corners=False,
        ).clamp(-1, 1).to(self.model.dtype)

        with torch.no_grad():
            z_hires = self.model.encode(img_up.to(self.model.device))

        z_refined = self.sampler.img2img(
            src_latent=z_hires,
            strength=hires_strength,
            uc=emb.uc,
            c=emb.c,
            cfg_guidance=cfg_guidance,
            model_kwargs=model_kwargs,
        )

        # 7. Final decode
        with torch.no_grad():
            img = self.model.decode(z_refined)
        return (img / 2 + 0.5).clamp(0, 1).detach().cpu()

    # ── SDXL conditioning builder ─────────────────────────────────────────────

    def _build_model_kwargs(
        self,
        emb: TextEmbedding,
        cfg_guidance: float,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int],
        crops_coords_top_left: Tuple[int, int],
        negative_original_size: Optional[Tuple[int, int]],
        negative_crops_coords_top_left: Tuple[int, int],
    ) -> dict:
        assert emb.pool_c is not None and emb.pool_uc is not None, \
            "SDXL pipeline requires pooled embeddings from SDXLTextEncoder"

        dtype = emb.c.dtype
        proj_dim = int(emb.pool_c.shape[-1])

        pos_ids = self._make_time_ids(original_size, crops_coords_top_left, target_size, dtype, proj_dim)
        if negative_original_size is not None:
            neg_ids = self._make_time_ids(negative_original_size, negative_crops_coords_top_left,
                                          target_size, dtype, proj_dim)
        else:
            neg_ids = pos_ids

        do_cfg = cfg_guidance not in (0.0, 1.0)
        if do_cfg:
            text_embeds = torch.cat([emb.pool_uc, emb.pool_c], dim=0)
            time_ids    = torch.cat([neg_ids, pos_ids], dim=0)
        else:
            text_embeds = emb.pool_c
            time_ids    = pos_ids

        return {
            "text_embeds": text_embeds.to(self.model.device),
            "time_ids":    time_ids.to(self.model.device),
        }

    def _make_time_ids(self, original_size, crops_coords_top_left, target_size, dtype, proj_dim) -> torch.Tensor:
        ids = list(original_size + crops_coords_top_left + target_size)
        expected = self.model.unet.add_embedding.linear_1.in_features
        got = self.model.unet.config.addition_time_embed_dim * len(ids) + proj_dim
        assert expected == got, (
            f"add_time_ids size mismatch: expected {expected}, got {got}. "
            "Check unet.config.addition_time_embed_dim and text_encoder_2 projection dim."
        )
        return torch.tensor([ids], dtype=dtype)
