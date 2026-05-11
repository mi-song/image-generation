from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class TextEmbedding:
    uc: torch.Tensor            # uncond hidden states  [1, seq, dim]
    c: torch.Tensor             # cond hidden states    [1, seq, dim]
    pool_uc: Optional[torch.Tensor] = None   # pooled uncond  [1, pool_dim]  (SDXL only)
    pool_c: Optional[torch.Tensor] = None    # pooled cond    [1, pool_dim]  (SDXL only)


class TextEncoderBase(ABC):
    @abstractmethod
    def encode(
        self,
        null_prompt: str,
        prompt: str,
        clip_skip: Optional[int] = None,
    ) -> TextEmbedding: ...
