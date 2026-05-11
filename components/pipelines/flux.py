from typing import Optional, Tuple

import numpy as np
import torch


class FluxPipelineWrapper:
    """
    Thin wrapper around diffusers FluxPipeline.
    Exposes the same .run() interface as SDXLPipeline so factory.py works uniformly.
    Flux uses Flow Matching — negative prompts and custom samplers are not applicable.
    """

    def __init__(
        self,
        model_key: str = "black-forest-labs/FLUX.1-dev",
        num_sampling: int = 20,
        device: str = "cuda",
        cpu_offload: str = "model",  # "none" | "model" | "sequential"
    ):
        from diffusers import FluxPipeline

        self._pipe = FluxPipeline.from_pretrained(model_key, torch_dtype=torch.bfloat16)
        if cpu_offload == "sequential":
            self._pipe.enable_sequential_cpu_offload()   # 최소 VRAM, 느림
        elif cpu_offload == "model":
            self._pipe.enable_model_cpu_offload()        # 균형
        else:
            self._pipe = self._pipe.to(device)           # 풀 VRAM

        # CLIP-L has a hard limit of 77 tokens — force truncation so it doesn't crash
        if self._pipe.tokenizer is not None:
            self._pipe.tokenizer.model_max_length = 77

        self._num_steps = num_sampling
        self.device = device

    def run(
        self,
        prompt: str,
        null_prompt: str = "",    # ignored — Flux has no negative prompt
        cfg_guidance: float = 3.5,
        target_size: Optional[Tuple[int, int]] = (1024, 1024),
        **kwargs,                 # absorb hires_fix, clip_skip, etc.
    ) -> torch.Tensor:
        H, W = target_size or (1024, 1024)

        images = self._pipe(
            prompt=prompt,
            num_inference_steps=self._num_steps,
            guidance_scale=cfg_guidance,
            height=H,
            width=W,
            max_sequence_length=512,  # T5 context length
        ).images

        img_np = np.array(images[0]).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
