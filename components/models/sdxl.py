import os
from typing import Any, Dict, Optional

import torch
from diffusers import (AutoencoderKL, DDIMScheduler, EulerDiscreteScheduler,
                       StableDiffusionXLPipeline)
from diffusers.models.attention_processor import (AttnProcessor2_0,
                                                  LoRAAttnProcessor2_0,
                                                  LoRAXFormersAttnProcessor,
                                                  XFormersAttnProcessor)

from .base import ModelBase, NoiseOutput

# LyCORIS key patterns that distinguish it from standard LoRA
_LYCORIS_PATTERNS = ("hada_w1", "lokr_w1", "glora_", "boft_", "oft_diag")


def _is_lycoris(path: str) -> bool:
    """Check if the weight file uses LyCORIS format by inspecting key names only (no tensor load)."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".safetensors":
            import safetensors
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
        else:
            keys = list(torch.load(path, map_location="cpu").keys())
        return any(any(p in k for p in _LYCORIS_PATTERNS) for k in keys)
    except Exception:
        return False


def _load_lycoris(pipe, lora_path: str, lora_scale: float) -> None:
    """Load LyCORIS weights, supporting both 1.x and 2.x API."""

    # lycoris 2.x API
    try:
        from lycoris import create_lycoris_from_weights
        for module in (pipe.unet, pipe.text_encoder, pipe.text_encoder_2):
            if module is None:
                continue
            try:
                net, _ = create_lycoris_from_weights(lora_scale, lora_path, module)
                net.merge_to()
            except Exception:
                pass
        print(f"[LyCORIS 2.x] loaded: {lora_path}")
        return
    except ImportError:
        pass

    # lycoris 1.x API
    try:
        from lycoris.kohya import create_network_from_weights
        network, weights_sd = create_network_from_weights(
            multiplier=lora_scale, file=lora_path, vae=None,
            text_encoder=pipe.text_encoder, unet=pipe.unet, for_inference=True,
        )
        network.merge_to(pipe.text_encoder, pipe.unet, weights_sd)
        print(f"[LyCORIS 1.x] loaded: {lora_path}")
        return
    except ImportError:
        pass

    raise ImportError(
        "lycoris-lora가 설치돼 있지만 API를 인식할 수 없습니다. "
        "pip install --upgrade lycoris-lora 로 업그레이드하세요."
    )


def _load_lora(pipe, lora_path: str, lora_scale: float) -> None:
    """Load LoRA or LyCORIS weights into pipe."""
    if _is_lycoris(lora_path):
        _load_lycoris(pipe, lora_path, lora_scale)
    else:
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)
        print(f"[LoRA] loaded: {lora_path}")


class SDXLModel(ModelBase):
    def __init__(
        self,
        model_key: str = "stabilityai/stable-diffusion-xl-base-1.0",
        num_sampling: int = 50,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        use_model_vae: bool = False,
        prediction_type: Optional[str] = None,
        timestep_spacing: Optional[str] = None,
    ):
        self.device = device
        self.dtype = dtype

        ext = os.path.splitext(model_key)[1].lower()
        if ext in (".safetensors", ".ckpt"):
            pipe = StableDiffusionXLPipeline.from_single_file(model_key, torch_dtype=dtype).to(device)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(model_key, torch_dtype=dtype).to(device)

        if lora_path:
            _load_lora(pipe, lora_path, lora_scale)

        if use_model_vae:
            self.vae = pipe.vae
        else:
            self.vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype
            ).to(device)

        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.unet = pipe.unet

        self.prediction_type = getattr(self.unet.config, "prediction_type", "epsilon")
        if prediction_type is not None:
            self.prediction_type = prediction_type

        self._pipe_scheduler_config = pipe.scheduler.config
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size

        _ts = timestep_spacing or getattr(pipe.scheduler.config, "timestep_spacing", "trailing")
        if ext in (".safetensors", ".ckpt"):
            sched = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing=_ts)
        else:
            sched = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", timestep_spacing=_ts)

        self._total_alphas = sched.alphas_cumprod.clone()
        self._sigmas = (1 - self._total_alphas).sqrt() / self._total_alphas.sqrt()
        self._log_sigmas = self._sigmas.log()

        N_ts = len(sched.timesteps)
        sched.set_timesteps(num_sampling, device=device)
        self._skip = N_ts // num_sampling
        self._final_alpha_cumprod = sched.final_alpha_cumprod.to(device)
        # prepend 1.0 for t=-1 lookup (DDIM convention)
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
    @property
    def scheduler_config(self): return self._pipe_scheduler_config

    # ── alpha lookup ──────────────────────────────────────────────────────────

    def alpha(self, t) -> torch.Tensor:
        t_val = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        if t_val < 0:
            return self._final_alpha_cumprod
        return self._total_alphas[t_val].to(self.device)

    # ── UNet ─────────────────────────────────────────────────────────────────

    def _unet_forward(self, zt, t, uc, c, model_kwargs):
        t_in = t.unsqueeze(0)
        kwargs = {"added_cond_kwargs": model_kwargs} if model_kwargs is not None else {}
        if uc is None:
            out_c = self.unet(zt, t_in, encoder_hidden_states=c, **kwargs)["sample"]
            return out_c, out_c
        if c is None:
            out_uc = self.unet(zt, t_in, encoder_hidden_states=uc, **kwargs)["sample"]
            return out_uc, out_uc
        c_embed = torch.cat([uc, c], dim=0)
        z_in = torch.cat([zt] * 2)
        t_in = torch.cat([t_in] * 2)
        out = self.unet(z_in, t_in, encoder_hidden_states=c_embed, **kwargs)["sample"]
        return out.chunk(2)

    def predict_noise(self, zt, t, uc, c, model_kwargs=None) -> NoiseOutput:
        noise_uc, noise_c = self._unet_forward(zt, t, uc, c, model_kwargs)
        if self.prediction_type == "v_prediction":
            t_idx = t.long().clamp(0, len(self._total_alphas) - 1)
            at = self._total_alphas[t_idx].to(zt.device)
            noise_uc = at.sqrt() * noise_uc + (1 - at).sqrt() * zt
            noise_c  = at.sqrt() * noise_c  + (1 - at).sqrt() * zt
        return NoiseOutput(noise_uc, noise_c)

    def predict_raw(self, zt, t, uc, c, model_kwargs=None) -> NoiseOutput:
        out_uc, out_c = self._unet_forward(zt, t, uc, c, model_kwargs)
        return NoiseOutput(out_uc, out_c)

    # ── VAE ──────────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor

    def decode(self, zt: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(zt / self.vae.config.scaling_factor).sample.float()


class SDXLLightningModel(SDXLModel):
    def __init__(
        self,
        model_ckpt: str,
        num_sampling: int = 4,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        # Reuse parent __init__ but with EulerDiscreteScheduler + trailing
        super().__init__(
            model_key=model_ckpt,
            num_sampling=num_sampling,
            dtype=dtype,
            device=device,
            use_model_vae=False,
            timestep_spacing="trailing",
        )
        # Override scheduler to EulerDiscrete (Lightning convention)
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_single_file(model_ckpt, torch_dtype=dtype).to(device)
        euler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        self._total_alphas = euler.alphas_cumprod.clone()
        self._sigmas = (1 - self._total_alphas).sqrt() / self._total_alphas.sqrt()
        self._log_sigmas = self._sigmas.log()
        N_ts = len(euler.timesteps)
        euler.set_timesteps(num_sampling, device=device)
        self._skip = N_ts // num_sampling
        euler.alphas_cumprod = torch.cat([torch.tensor([1.0]), euler.alphas_cumprod]).to(device)
        self._scheduler = euler
