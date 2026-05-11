import sys
sys.path.insert(0, ".")

import random
import numpy as np
import torch
import gradio as gr
from PIL import Image

from factory import build_pipeline
from utils.log_util import set_seed

_pipe = None
_pipe_cfg = {}


def _need_reload(cfg: dict) -> bool:
    keys = ["model_type", "model_key", "prediction_type", "lora_path",
            "lora_scale", "use_model_vae", "timestep_spacing", "use_compel"]
    return any(_pipe_cfg.get(k) != cfg[k] for k in keys)


def generate(
    model_type, model_key, prediction_type, lora_path, lora_scale,
    use_model_vae, use_compel, timestep_spacing,
    sampler, schedule, nfe, cfg,
    height, width,
    prompt, null_prompt,
    seed, random_seed,
    hires_fix, hires_scale, hires_strength,
    clip_skip,
    flux_cpu_offload,
):
    global _pipe, _pipe_cfg

    if random_seed:
        seed = random.randint(0, 2**32 - 1)

    is_flux = (model_type == "flux")
    prediction_type = prediction_type if prediction_type != "auto" else None
    timestep_spacing = timestep_spacing if timestep_spacing != "auto" else None
    lora_path = lora_path.strip() or None

    cfg_now = dict(
        model_type=model_type, model_key=model_key.strip(),
        prediction_type=prediction_type, lora_path=lora_path,
        lora_scale=lora_scale, use_model_vae=use_model_vae,
        timestep_spacing=timestep_spacing, use_compel=use_compel,
        flux_cpu_offload=flux_cpu_offload,
    )

    if _pipe is None or _need_reload(cfg_now):
        yield None, "⏳ 모델 로딩 중... (처음 한 번만 시간이 걸립니다)", seed
        _pipe = build_pipeline(
            model_type=model_type,
            sampler_name=sampler,
            model_key=model_key.strip(),
            num_sampling=int(nfe),
            device="cuda",
            lora_path=lora_path,
            lora_scale=lora_scale,
            use_model_vae=use_model_vae,
            prediction_type=prediction_type,
            timestep_spacing=timestep_spacing,
            use_compel=use_compel,
            schedule=schedule,
            flux_cpu_offload=flux_cpu_offload,
        )
        _pipe_cfg = cfg_now
    elif not is_flux:
        from components.samplers import get_sampler_class
        sampler_cls = get_sampler_class(sampler)
        _pipe.sampler = sampler_cls(_pipe.model, schedule=schedule)
        _pipe.model.scheduler.set_timesteps(int(nfe))

    set_seed(int(seed))
    yield None, f"🎨 이미지 생성 중... (seed={seed})", seed

    clip_skip_val = int(clip_skip) if int(clip_skip) > 1 else None

    try:
        result = _pipe.run(
            null_prompt=null_prompt.strip(),
            prompt=prompt.strip(),
            cfg_guidance=float(cfg),
            target_size=(int(height), int(width)),
            hires_fix=hires_fix,
            hires_scale=float(hires_scale),
            hires_strength=float(hires_strength),
            clip_skip=clip_skip_val,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        yield None, "❌ VRAM 부족 — 크기를 줄이거나 고급 설정에서 Flux Offload를 켜세요.", seed
        return
    except Exception as e:
        yield None, f"❌ 오류: {type(e).__name__}: {e}", seed
        return

    img_np = (result.squeeze(0).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    yield pil_img, f"✅ 완료! (seed={seed})", seed


# ── 상수 ──────────────────────────────────────────────────────────────────────

DEFAULT_PROMPT = (
    "masterpiece, best quality, newest, absurdres, highres, chibi, 1girl, "
    "solo, twintails, thighhighs, purple eyes, hat, long hair, smile, shorts, jacket, "
    "jacket on shoulders, white thighhighs, boots, full body, simple background, "
    "looking at viewer, grey background, hair ornament, blonde hair, black jacket, "
    "rabbit hair ornament, very long hair, uniform, military uniform, military, "
    "black footwear, white hair"
)
DEFAULT_NEG = (
    "worst quality, old, early, low quality, lowres, signature, username, logo, "
    "bad hands, mutated hands, mammal, anthro, furry, ambiguous form, feral, semi-anthro"
)

_MODEL_DEFAULTS = {
    "sdxl": "C:/Users/82103/Downloads/noobaiXLNAIXL_vPred05Version.safetensors",
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sd20": "stabilityai/stable-diffusion-2-1",
    "flux": "black-forest-labs/FLUX.1-dev",
}
_CFG_DEFAULTS     = {"sdxl": 1.0,  "sd15": 7.0,  "sd20": 7.0,  "flux": 3.5}
_NFE_DEFAULTS     = {"sdxl": 30,   "sd15": 30,   "sd20": 30,   "flux": 20}
_H_DEFAULTS       = {"sdxl": 1216, "sd15": 512,  "sd20": 768,  "flux": 1024}
_W_DEFAULTS       = {"sdxl": 832,  "sd15": 512,  "sd20": 768,  "flux": 1024}
_PRED_DEFAULTS    = {"sdxl": "v_prediction", "sd15": "epsilon", "sd20": "v_prediction", "flux": "auto"}
_SAMPLER_DEFAULTS = {"sdxl": "dpm++_2m_cfg++", "sd15": "dpm++_2m_cfg++", "sd20": "dpm++_2m_cfg++", "flux": "euler"}

SIZE_PRESETS = {
    "세로형 (832×1216)":   (832,  1216),
    "정사각형 (1024×1024)": (1024, 1024),
    "가로형 (1216×832)":   (1216, 832),
    "소형 (512×512)":      (512,  512),
}

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }

/* ── 헤더 ── */
#app-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 14px; padding: 18px 28px; margin-bottom: 18px; text-align: center;
}
#app-header h1 { font-size: 1.65rem; font-weight: 700; margin: 0 0 3px; color: white; letter-spacing: -0.02em; }
#app-header p  { font-size: 0.82rem; opacity: 0.82; margin: 0; color: white; }

/* ── 스텝 헤더 ── */
.s-hdr {
    display: flex; align-items: center; gap: 9px;
    padding: 14px 0 8px;
    border-top: 1px solid var(--border-color-primary);
}
.s-hdr-first { display: flex; align-items: center; gap: 9px; padding-bottom: 8px; }
.s-dot {
    width: 22px; height: 22px; border-radius: 50%; flex-shrink: 0;
    background: linear-gradient(135deg, #667eea, #764ba2);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.68rem; font-weight: 800; color: white;
}
.s-label {
    font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.09em; color: #888;
}

/* ── 생성 버튼 ── */
#gen-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important; border-radius: 10px !important;
    font-size: 1.08rem !important; font-weight: 700 !important;
    padding: 14px !important; color: white !important;
    transition: filter 0.15s, transform 0.1s !important;
}
#gen-btn:hover { filter: brightness(1.1) !important; transform: translateY(-1px) !important; }

/* ── 결과 이미지 ── */
#output-img { border-radius: 12px !important; overflow: hidden !important; }

/* ── 상태 ── */
#status-box textarea { font-family: 'Consolas', monospace !important; font-size: 0.85rem !important; }

/* ── 오른쪽 컬럼 상단 고정 ── */
#right-col { align-self: flex-start !important; position: sticky !important; top: 12px !important; }
"""


def S(n, label, first=False):
    cls = "s-hdr-first" if first else "s-hdr"
    return gr.HTML(
        f'<div class="{cls}">'
        f'<div class="s-dot">{n}</div>'
        f'<div class="s-label">{label}</div>'
        f'</div>'
    )


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="CFGpp Image Generator") as demo:

    gr.HTML("""
    <div id="app-header">
        <h1>Image Generator</h1>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── 왼쪽: 설정 단계 ─────────────────────────────────────────────────
        with gr.Column(scale=4, min_width=360):

            S(1, "모델 선택", first=True)
            model_type = gr.Dropdown(
                choices=[
                    ("SDXL",   "sdxl"),
                    ("SD 1.5", "sd15"),
                    ("SD 2.0", "sd20"),
                    ("Flux",   "flux"),
                ],
                value="sdxl", label="모델 종류",
            )
            model_file = gr.File(
                label="모델 파일 드래그 또는 클릭  (.safetensors / .ckpt)",
                file_types=[".safetensors", ".ckpt", ".pt"], type="filepath",
            )
            model_key = gr.State(value=_MODEL_DEFAULTS["sdxl"])

            with gr.Accordion("＋  LoRA 추가  (선택)", open=False):
                lora_file  = gr.File(
                    label="LoRA 파일 드래그 또는 클릭  (.safetensors)",
                    file_types=[".safetensors", ".pt"], type="filepath",
                )
                lora_path  = gr.Textbox(value="", label="경로 직접 입력",
                                        placeholder="비워두면 LoRA 미사용")
                lora_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05,
                                       label="적용 강도",
                                       info="0 = 끔  /  1 = 기본  /  2 = 최대")

            S(2, "이미지 크기")
            size_preset = gr.Radio(
                choices=list(SIZE_PRESETS.keys()),
                value="세로형 (832×1216)", label="비율",
            )
            with gr.Row():
                width  = gr.Number(value=832,  label="가로 px", precision=0, minimum=64, maximum=4096)
                height = gr.Number(value=1216, label="세로 px", precision=0, minimum=64, maximum=4096)

            S(3, "생성 설정")
            with gr.Row():
                nfe = gr.Slider(1, 100, value=30, step=1,
                    label="스텝 수", info="20~35 권장")
                cfg = gr.Slider(1.0, 20.0, value=1.0, step=0.5,
                    label="CFG 강도", info="SDXL/CFG++ → 1~3  /  SD → 7 전후")
            with gr.Row():
                sampler = gr.Dropdown(
                    choices=[
                        ("DPM++ 2M CFG++ (권장)", "dpm++_2m_cfg++"),
                        ("Euler CFG++",           "euler_cfg++"),
                        ("Euler Ancestral",       "euler_a"),
                        ("Euler",                 "euler"),
                        ("DDIM CFG++",            "ddim_cfg++"),
                        ("DDIM",                  "ddim"),
                        ("Euler A Native",        "euler_a_native"),
                    ],
                    value="dpm++_2m_cfg++", label="샘플러",
                    info="이미지를 완성해 나가는 알고리즘", scale=3,
                )
                schedule = gr.Dropdown(
                    choices=[
                        ("Karras (권장)",          "karras"),
                        ("Linear",                 "linear"),
                        ("Exponential",            "exponential"),
                        ("Align Your Steps (AYS)", "ays"),
                    ],
                    value="karras", label="스케줄",
                    info="각 스텝의 노이즈 제거 곡선", scale=2,
                )

            S(4, "시드")
            with gr.Row():
                seed        = gr.Number(value=338732427, label="시드 번호",
                                        precision=0, info="같은 번호 = 동일한 결과 재현", scale=3)
                random_seed = gr.Checkbox(value=False, label="매번 랜덤", scale=1)

            with gr.Accordion("⚙️  고급 설정  (선택)", open=False):
                gr.Markdown("기본값으로도 잘 동작합니다.")

                gr.HTML('<div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#a0aec0;border-top:1px solid var(--border-color-primary);padding-top:8px;margin:10px 0 4px">Hires Fix — 해상도 향상</div>')
                gr.Markdown("저해상도 초안 생성 후 업스케일. 품질↑ 속도↓")
                hires_fix = gr.Checkbox(value=False, label="켜기")
                with gr.Row():
                    hires_scale    = gr.Slider(0.4, 0.9, value=0.75, step=0.05,
                                               label="1차 배율", info="낮을수록 빠름")
                    hires_strength = gr.Slider(0.1, 1.0, value=0.5, step=0.05,
                                               label="변화 강도", info="높을수록 초안에서 많이 바뀜")

                gr.HTML('<div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#a0aec0;border-top:1px solid var(--border-color-primary);padding-top:8px;margin:10px 0 4px">예측 & 샘플링</div>')
                with gr.Row():
                    prediction_type = gr.Dropdown(
                        choices=["auto", "epsilon", "v_prediction"],
                        value="v_prediction", label="예측 방식",
                        info="v-pred → v_prediction  /  나머지 → epsilon  /  모르면 auto",
                        scale=2,
                    )
                    clip_skip = gr.Slider(1, 2, value=1, step=1,
                        label="CLIP Skip", info="애니 전용 모델은 2 권장", scale=1)
                with gr.Row():
                    timestep_spacing = gr.Dropdown(
                        choices=["auto", "trailing", "linspace", "leading"],
                        value="auto", label="타임스텝 간격",
                        info="거의 변경 불필요", scale=2,
                    )
                    flux_cpu_offload = gr.Dropdown(
                        choices=["none", "model", "sequential"],
                        value="model", label="Flux VRAM 절약",
                        info="Flux 전용. sequential = VRAM 최소화", scale=1,
                    )

                gr.HTML('<div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#a0aec0;border-top:1px solid var(--border-color-primary);padding-top:8px;margin:10px 0 4px">기타</div>')
                with gr.Row():
                    use_model_vae = gr.Checkbox(value=True, label="모델 내장 VAE 사용")
                    use_compel    = gr.Checkbox(value=True, label="77토큰 초과 프롬프트 지원")

        # ── 오른쪽: 프롬프트 + 결과 ─────────────────────────────────────────
        with gr.Column(scale=6, min_width=480, elem_id="right-col"):

            prompt = gr.Textbox(
                value=DEFAULT_PROMPT,
                label="프롬프트",
                info="원하는 이미지를 영어로 묘사하세요",
                lines=5, max_lines=14,
                placeholder="예) masterpiece, best quality, 1girl, smile, blue sky...",
            )
            null_prompt = gr.Textbox(
                value=DEFAULT_NEG,
                label="네거티브 프롬프트",
                info="원하지 않는 요소를 입력하세요",
                lines=3, max_lines=8,
                placeholder="예) worst quality, blurry, bad hands...",
            )

            generate_btn = gr.Button(
                "🚀  이미지 생성", variant="primary", size="lg", elem_id="gen-btn",
            )

            output_image = gr.Image(
                label="생성된 이미지", type="pil",
                elem_id="output-img", height=600,
            )

            with gr.Row():
                status_text = gr.Textbox(
                    label="상태", interactive=False,
                    elem_id="status-box", scale=4,
                )
                out_seed = gr.Number(label="사용된 시드", interactive=False, scale=1)

    # ── 이벤트 핸들러 ─────────────────────────────────────────────────────────

    def on_model_file_drop(f):
        if f is None:
            return gr.update()
        return f if isinstance(f, str) else getattr(f, "name", str(f))

    def on_lora_file_drop(f):
        if f is None:
            return gr.update()
        path = f if isinstance(f, str) else getattr(f, "name", str(f))
        return gr.update(value=path)

    model_file.change(fn=on_model_file_drop, inputs=[model_file], outputs=[model_key])
    lora_file.change(fn=on_lora_file_drop,   inputs=[lora_file],  outputs=[lora_path])

    def on_model_type_change(mt):
        return (
            _MODEL_DEFAULTS.get(mt, ""),
            gr.update(value=_CFG_DEFAULTS.get(mt, 7.0)),
            gr.update(value=_NFE_DEFAULTS.get(mt, 30)),
            gr.update(value=_H_DEFAULTS.get(mt, 1024)),
            gr.update(value=_W_DEFAULTS.get(mt, 1024)),
            gr.update(value=_PRED_DEFAULTS.get(mt, "auto")),
            gr.update(value=_SAMPLER_DEFAULTS.get(mt, "dpm++_2m_cfg++")),
        )

    model_type.change(
        fn=on_model_type_change,
        inputs=[model_type],
        outputs=[model_key, cfg, nfe, height, width, prediction_type, sampler],
    )

    def on_size_preset(preset):
        w, h = SIZE_PRESETS.get(preset, (1024, 1024))
        return gr.update(value=w), gr.update(value=h)

    size_preset.change(fn=on_size_preset, inputs=[size_preset], outputs=[width, height])

    generate_btn.click(
        fn=generate,
        inputs=[
            model_type, model_key, prediction_type, lora_path, lora_scale,
            use_model_vae, use_compel, timestep_spacing,
            sampler, schedule, nfe, cfg,
            height, width,
            prompt, null_prompt,
            seed, random_seed,
            hires_fix, hires_scale, hires_strength,
            clip_skip,
            flux_cpu_offload,
        ],
        outputs=[output_image, status_text, out_seed],
    )


if __name__ == "__main__":
    demo.launch(share=True, inbrowser=True, theme=gr.themes.Soft(), css=CSS)
