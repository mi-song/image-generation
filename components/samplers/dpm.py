from typing import Any, Callable, Dict, Optional, Tuple

import torch
from tqdm import tqdm

from .base import SamplerBase
from components.samplers import register_sampler


@register_sampler("dpm++_2m_cfg++")
class DPMpp2mCFGppSampler(SamplerBase):
    @torch.autocast("cuda")
    def sample(self, uc, c, cfg_guidance, latent_shape, model_kwargs=None,
               src_latent=None, callback_fn=None) -> torch.Tensor:
        t_fn = lambda sigma: sigma.log().neg()

        sigmas = self._get_sigmas()
        x = self.random_latent(latent_shape).to(torch.float16) * sigmas[0]
        old_denoised = None

        for i in tqdm(range(len(sigmas) - 1), desc=f"dpm++_2m_cfg++ [{self.schedule}]"):
            sigma = sigmas[i]
            t = self.model.timestep(sigma)
            # alpha from sigma: sigma = sqrt((1-a)/a) → a = 1/(sigma^2+1)
            at = (1.0 / (sigma ** 2 + 1)).to(torch.float16)
            c_in, c_out = at.sqrt(), -sigma

            with torch.no_grad():
                out = self.model.predict_noise(x * c_in, t, uc, c, model_kwargs)
                noise_pred = out.noise_uc + cfg_guidance * (out.noise_c - out.noise_uc)

            denoised        = x + c_out * noise_pred
            uncond_denoised = x + c_out * out.noise_uc

            t_log, t_next_log = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next_log - t_log
            if old_denoised is None or sigmas[i + 1] == 0:
                x = denoised + self.model.to_d(x, sigmas[i], uncond_denoised) * sigmas[i + 1]
            else:
                h_last = t_log - t_fn(sigmas[i - 1])
                r = h_last / h
                extra1 = -torch.exp(-h) * uncond_denoised - (-h).expm1() * (uncond_denoised - old_denoised) / (2 * r)
                x = denoised + extra1 + torch.exp(-h) * x
            old_denoised = uncond_denoised

            if callback_fn:
                kw = callback_fn(i, t, {"z0t": denoised.detach(), "zt": x.detach(), "decode": self.model.decode})
                denoised, x = kw["z0t"], kw["zt"]

        return x
