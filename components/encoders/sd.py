from typing import Optional

import torch

from .base import TextEmbedding, TextEncoderBase


class SDTextEncoder(TextEncoderBase):
    def __init__(self, tokenizer, text_encoder, device: str = "cuda", use_compel: bool = False):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.compel = None

        if use_compel:
            try:
                from compel import Compel
                self.compel = Compel(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    truncate_long_prompts=False,
                )
                print("Compel enabled: unlimited prompt length")
            except ImportError:
                print("compel not installed — pip install compel")

    @torch.no_grad()
    def encode(self, null_prompt: str, prompt: str, clip_skip: Optional[int] = None) -> TextEmbedding:
        if self.compel is not None:
            c  = self.compel(prompt  if isinstance(prompt, str)       else prompt[0])
            uc = self.compel(null_prompt if isinstance(null_prompt, str) else null_prompt[0])
            max_len = max(c.shape[1], uc.shape[1])
            c, uc = self._pad(c, max_len), self._pad(uc, max_len)
            return TextEmbedding(uc=uc, c=c)

        null_in = self.tokenizer(null_prompt, padding="max_length",
                                  max_length=self.tokenizer.model_max_length,
                                  return_tensors="pt")
        uc = self.text_encoder(null_in.input_ids.to(self.device))[0]

        text_in = self.tokenizer(prompt, padding="max_length",
                                  max_length=self.tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")
        c = self.text_encoder(text_in.input_ids.to(self.device))[0]
        return TextEmbedding(uc=uc, c=c)

    @staticmethod
    def _pad(t: torch.Tensor, target_len: int) -> torch.Tensor:
        if t.shape[1] >= target_len:
            return t
        pad = torch.zeros((t.shape[0], target_len - t.shape[1], t.shape[2]),
                           device=t.device, dtype=t.dtype)
        return torch.cat([t, pad], dim=1)
