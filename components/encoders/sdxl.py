from typing import Optional, Tuple

import torch

from .base import TextEmbedding, TextEncoderBase


class SDXLTextEncoder(TextEncoderBase):
    def __init__(
        self,
        tokenizer_1,
        tokenizer_2,
        text_enc_1,
        text_enc_2,
        device: str = "cuda",
        use_compel: bool = False,
    ):
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.text_enc_1 = text_enc_1
        self.text_enc_2 = text_enc_2
        self.device = device
        self.compel = None

        if use_compel:
            try:
                from compel import Compel, ReturnedEmbeddingsType
                self.compel = Compel(
                    tokenizer=[tokenizer_1, tokenizer_2],
                    text_encoder=[text_enc_1, text_enc_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False,
                )
                print("Compel enabled: unlimited prompt length")
            except ImportError:
                print("compel not installed — pip install compel")

    @torch.no_grad()
    def encode(
        self,
        null_prompt: str,
        prompt: str,
        null_prompt_2: Optional[str] = None,
        prompt_2: Optional[str] = None,
        clip_skip: Optional[int] = None,
    ) -> TextEmbedding:
        if self.compel is not None:
            return self._encode_compel(null_prompt, prompt)
        return self._encode_manual(null_prompt, prompt, null_prompt_2, prompt_2, clip_skip)

    def _encode_manual(self, null_prompt, prompt, null_prompt_2, prompt_2, clip_skip) -> TextEmbedding:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        null_prompt = [null_prompt] if isinstance(null_prompt, str) else null_prompt

        emb1, pool_c = self._embed(prompt, self.tokenizer_1, self.text_enc_1, clip_skip)
        null1, pool_uc = self._embed(null_prompt, self.tokenizer_1, self.text_enc_1, clip_skip)

        if prompt_2 is not None:
            emb2, pool_c = self._embed([prompt_2] if isinstance(prompt_2, str) else prompt_2,
                                        self.tokenizer_2, self.text_enc_2, clip_skip)
            c = torch.cat([emb1, emb2], dim=-1)
        else:
            c = emb1

        if null_prompt_2 is not None:
            null2, pool_uc = self._embed([null_prompt_2] if isinstance(null_prompt_2, str) else null_prompt_2,
                                          self.tokenizer_2, self.text_enc_2, clip_skip)
            uc = torch.cat([null1, null2], dim=-1)
        else:
            uc = null1

        return TextEmbedding(uc=uc, c=c, pool_uc=pool_uc, pool_c=pool_c)

    def _embed(self, prompt, tokenizer, text_enc, clip_skip) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        out = text_enc(inputs.input_ids.to(self.device), output_hidden_states=True)
        pool = out[0]
        hidden = out.hidden_states[-2] if clip_skip is None else out.hidden_states[-(clip_skip + 2)]
        return hidden, pool

    def _encode_compel(self, null_prompt: str, prompt: str) -> TextEmbedding:
        c, pool_c = self.compel(prompt if isinstance(prompt, str) else prompt[0])
        uc, pool_uc = self.compel(null_prompt if isinstance(null_prompt, str) else null_prompt[0])
        # pad to same sequence length
        max_len = max(c.shape[1], uc.shape[1])
        c  = self._pad(c,  max_len)
        uc = self._pad(uc, max_len)
        return TextEmbedding(uc=uc, c=c, pool_uc=pool_uc, pool_c=pool_c)

    @staticmethod
    def _pad(t: torch.Tensor, target_len: int) -> torch.Tensor:
        if t.shape[1] >= target_len:
            return t
        pad = torch.zeros((t.shape[0], target_len - t.shape[1], t.shape[2]),
                           device=t.device, dtype=t.dtype)
        return torch.cat([t, pad], dim=1)
