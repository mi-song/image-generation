from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class NoiseOutput:
    noise_uc: torch.Tensor
    noise_c: torch.Tensor


class ModelBase(ABC):
    """
    UNet + VAE + scheduler alpha table.
    Text encoding and sampling logic live outside.
    """

    # ── UNet ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def predict_noise(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        uc: torch.Tensor,
        c: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> NoiseOutput:
        """Returns epsilon-space noise (v→ε conversion applied internally for v-pred)."""
        ...

    @abstractmethod
    def predict_raw(
        self,
        zt: torch.Tensor,
        t: torch.Tensor,
        uc: torch.Tensor,
        c: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> NoiseOutput:
        """Returns raw UNet output without any conversion. For use with scheduler.step()."""
        ...

    # ── Alpha / sigma helpers ─────────────────────────────────────────────────

    @abstractmethod
    def alpha(self, t) -> torch.Tensor:
        """total_alphas[t] lookup."""
        ...

    @property
    @abstractmethod
    def total_alphas(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def sigmas(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def log_sigmas(self) -> torch.Tensor: ...

    # ── Scheduler ────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def timesteps(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def skip(self) -> int: ...

    @property
    def scheduler_config(self) -> Optional[Dict[str, Any]]:
        """Original pipe scheduler config. Override in subclasses that support it."""
        return None

    # ── VAE ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, zt: torch.Tensor) -> torch.Tensor: ...

    # ── k-diffusion helpers (shared implementation) ──────────────────────────

    def calculate_input(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return x / (sigma ** 2 + 1) ** 0.5

    def calculate_denoised(self, x: torch.Tensor, model_pred: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return x - model_pred * sigma

    def to_d(self, x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        return (x - denoised) / sigma.item()

    def timestep(self, sigma: torch.Tensor) -> torch.Tensor:
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def kdiffusion_zt_to_denoised(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        uc: torch.Tensor,
        c: torch.Tensor,
        cfg_guidance: float,
        t: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (denoised, uncond_denoised). Used by all k-diffusion samplers."""
        xc = self.calculate_input(x, sigma)
        out = self.predict_noise(xc, t, uc, c, model_kwargs)
        noise_pred = out.noise_uc + cfg_guidance * (out.noise_c - out.noise_uc)
        denoised = self.calculate_denoised(x, noise_pred, sigma)
        uncond_denoised = self.calculate_denoised(x, out.noise_uc, sigma)
        return denoised, uncond_denoised
