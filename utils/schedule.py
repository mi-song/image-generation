import torch
import numpy as np

# Precomputed optimal timesteps from "Align Your Steps" (Sabour et al., 2024)
_AYS_TIMESTEPS = {
    "sdxl": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13],
    "sd":   [999, 850, 736, 645, 545, 455, 343, 233, 124, 24],
}


def get_sigmas_ays(n: int, total_alphas: torch.Tensor, is_sdxl: bool = True, device: str = "cpu") -> torch.Tensor:
    """
    Align Your Steps sigma schedule.
    For n != 10, interpolates the AYS curve in log-sigma space.
    """
    key = "sdxl" if is_sdxl else "sd"
    base_ts = torch.tensor(_AYS_TIMESTEPS[key], dtype=torch.long)
    base_alphas = total_alphas[base_ts]
    base_sigmas = ((1 - base_alphas) / base_alphas).sqrt().float()

    if n == len(base_ts):
        sigmas = base_sigmas
    else:
        # interpolate in log-sigma space along the AYS curve
        t_base = np.linspace(0, 1, len(base_ts))
        t_new  = np.linspace(0, 1, n)
        log_interp = np.interp(t_new, t_base, base_sigmas.log().numpy())
        sigmas = torch.from_numpy(np.exp(log_interp)).float()

    return torch.cat([sigmas.to(device), torch.zeros(1, device=device)])


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    ramp = torch.linspace(0, 1, n + 1, device=device)[:-1]
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])]).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    sigmas = torch.linspace(sigma_max.log(), sigma_min.log(), n, device=device).exp()
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])
