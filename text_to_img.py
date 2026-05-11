import sys
sys.path.insert(0, ".")

from pathlib import Path
from torchvision.utils import save_image

from factory import build_pipeline
from utils.log_util import set_seed

# ────────────────────────────────────────────
# Config
# ────────────────────────────────────────────
DEVICE    = "cuda"
SEED      = 338732427

MODEL_TYPE      = "sdxl"         # "sd15" | "sd20" | "sdxl"
MODEL_KEY       = "C:/Users/82103/Downloads/noobaiXLNAIXL_vPred05Version.safetensors"
PREDICTION_TYPE = "v_prediction" # None=자동감지 | "epsilon" | "v_prediction"
LORA_PATH       = "C:/Users/82103/Downloads/maplestory-000011prodigy.safetensors"
LORA_SCALE      = 1.0
USE_MODEL_VAE   = True
USE_COMPEL      = True           # pip install compel
TIMESTEP_SPACING = None          # None=모델 기본값 | "trailing" | "linspace" | "leading"
SAMPLER         = "dpm++_2m_cfg++"  # ddim | ddim_cfg++ | euler | euler_a | euler_a_native | euler_cfg++ | dpm++_2m_cfg++
SCHEDULE        = "karras"          # linear | karras | exponential  (ddim/euler_a_native은 무시)
NFE             = 30
CFG             = 1
TARGET_SIZE     = (1216, 832)    # (H, W)

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
OUTFILE = OUTDIR / "generated.png"
# ────────────────────────────────────────────


def main():
    set_seed(SEED)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(
        model_type=MODEL_TYPE,
        sampler_name=SAMPLER,
        model_key=MODEL_KEY,
        num_sampling=NFE,
        device=DEVICE,
        lora_path=LORA_PATH,
        lora_scale=LORA_SCALE,
        use_model_vae=USE_MODEL_VAE,
        prediction_type=PREDICTION_TYPE,
        timestep_spacing=TIMESTEP_SPACING,
        use_compel=USE_COMPEL,
        schedule=SCHEDULE,
    )

    print(f"prediction_type: {pipe.model.prediction_type}")

    result = pipe.run(
        null_prompt=NULL_PROMPT.strip(),
        prompt=PROMPT.strip(),
        cfg_guidance=CFG,
        target_size=TARGET_SIZE,
    )

    save_image(result, OUTFILE, normalize=False)
    print(f"saved → {OUTFILE}")


if __name__ == "__main__":
    main()
