"""
Pure diffusers reference script — no custom sampling code.
Uses Compel for unlimited prompt length (>77 tokens).
"""
import sys
sys.path.insert(0, ".")

import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from compel import Compel, ReturnedEmbeddingsType
from utils.log_util import set_seed

# ── settings ─────────────────────────────────────────────────────────────────
DEVICE          = "cuda"
SEED            = 338732427
MODEL_KEY       = "C:/Users/82103/Downloads/noobaiXLNAIXL_vPred05Version.safetensors"
LORA_PATH       = "C:/Users/82103/Downloads/maplestory-000011prodigy.safetensors"
LORA_SCALE      = 1.0
NFE             = 30
CFG             = 5
WIDTH, HEIGHT   = 832, 1216
PROMPT = """
masterpiece, best quality, newest, absurdres, highres,chibi, 1girl,
solo, twintails, thighhighs, purple eyes, hat, long hair, smile, shorts, jacket, jacket on shoulders,
white thighhighs, boots, full body, simple background, looking at viewer, grey background, hair ornament,
blonde hair, black jacket, rabbit hair ornament, very long hair, uniform, military uniform, military,
black footwear, white hair
"""
NULL_PROMPT = """
worst quality, old, early, low quality, lowres, signature, username, logo,
bad hands, mutated hands, mammal, anthro, furry, ambiguous form, feral, semi-anthro
"""
OUTDIR  = Path("outputs")
OUTFILE = OUTDIR / "reference_diffusers.png"
# ─────────────────────────────────────────────────────────────────────────────

def pad_to_same_length(tensors):
    max_len = max(t.shape[1] for t in tensors)
    padded = []
    for t in tensors:
        pad_len = max_len - t.shape[1]
        if pad_len > 0:
            pad = torch.zeros(
                (t.shape[0], pad_len, t.shape[2]),
                device=t.device, dtype=t.dtype
            )
            t = torch.cat([t, pad], dim=1)
        padded.append(t)
    return padded


def main():
    set_seed(SEED)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        MODEL_KEY, torch_dtype=torch.float16
    ).to(DEVICE)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config,
        prediction_type="v_prediction",
    )
    print(f"scheduler prediction_type: {pipe.scheduler.config.prediction_type}")

    if LORA_PATH:
        print("Loading LoRA...")
        pipe.load_lora_weights(LORA_PATH)
        pipe.fuse_lora(lora_scale=LORA_SCALE)

    # Compel: handles >77-token prompts for SDXL dual encoder
    compel = Compel(
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False,
    )

    conditioning, pooled = compel(PROMPT.strip())
    negative_conditioning, negative_pooled = compel(NULL_PROMPT.strip())
    conditioning, negative_conditioning = pad_to_same_length([conditioning, negative_conditioning])

    print("Generating...")
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    result = pipe(
        prompt_embeds=conditioning,
        pooled_prompt_embeds=pooled,
        negative_prompt_embeds=negative_conditioning,
        negative_pooled_prompt_embeds=negative_pooled,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NFE,
        guidance_scale=CFG,
        generator=generator,
    ).images[0]

    result.save(OUTFILE)
    print(f"saved → {OUTFILE}")

if __name__ == "__main__":
    main()
