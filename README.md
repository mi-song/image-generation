# Image Generation

로컬에서 동작하는 Stable Diffusion 계열 text-to-image WebUI - Gradio 기반

<br>

## 지원 모델

- **SDXL** (NoobAI 등 v-prediction 계열 포함)
- **SD 1.5**
- **SD 2.0**
- **Flux**

`.safetensors`, `.ckpt`, `.pt` 형식의 로컬 모델 파일을 드래그앤드롭으로 로드한다. HuggingFace repo ID도 직접 입력 가능.

<br>

## 주요 기능

- 다양한 샘플러 — DPM++ 2M CFG++, Euler CFG++, Euler Ancestral, Euler, DDIM / DDIM CFG++, Euler A Native
- 스케줄 선택 — Karras, Linear, Exponential, AYS (Align Your Steps)
- **LoRA** 로드 및 강도 조절
- **Compel**을 이용한 77토큰 초과 프롬프트 지원
- **Hires Fix** — 저해상도 초안 생성 후 업스케일
- **CLIP Skip** (애니메·일러스트 모델용)
- v-prediction / epsilon 예측 방식 선택
- Flux 전용 CPU offload (VRAM 절약)
- 시드 고정 / 랜덤
- 비율 프리셋 (세로형 / 정사각형 / 가로형 / 소형)

<br>

## 설치

```bash
git clone https://github.com/mi-song/image-generation.git
cd image-generation
pip install torch diffusers transformers gradio compel safetensors pillow numpy
```

CUDA 환경 권장.

<br>

## 실행

```bash
python app.py
```

기본적으로 Gradio가 `http://localhost:7860`에서 실행되고, `share=True` 옵션으로 임시 공개 링크도 함께 생성된다.

<br>

## 사용 흐름

1. **모델 종류 선택** — SDXL / SD 1.5 / SD 2.0 / Flux
2. **모델 파일 로드** — 로컬 파일 드래그 또는 HuggingFace 경로 입력
3. (선택) **LoRA 추가** — 강도 슬라이더로 조절
4. **이미지 크기 설정** — 프리셋 또는 직접 입력
5. **생성 설정** — 스텝 수, CFG 강도, 샘플러, 스케줄
6. **프롬프트 / 네거티브 프롬프트 입력**
7. **생성 버튼**

<br>

## 옵션 설명

### 예측 방식 (Prediction Type)

- `v_prediction` — SDXL v-pred 모델 (NoobAI v-pred 등), SD 2.0
- `epsilon` — SD 1.5, 대부분의 일반 모델
- `auto` — 모델 설정에서 자동 감지

### 77토큰 초과 프롬프트 지원

체크하면 Compel을 통해 75토큰을 넘는 프롬프트도 정상 처리. 청크 분할 + 임베딩 concat 방식.

### Hires Fix

저해상도(예: 75% 크기)로 먼저 생성한 뒤 목표 크기로 업스케일. 디테일 품질이 높아지는 대신 생성 시간이 늘어난다.

- **1차 배율** — 초안 크기. 낮을수록 빠름
- **변화 강도** — 업스케일 단계에서 초안에서 얼마나 벗어날지

### CLIP Skip

- `1` — 기본
- `2` — 마지막에서 두 번째 레이어의 hidden state 사용. NovelAI 시절부터 애니메·일러스트 모델에서 표준으로 쓰임

### Flux VRAM 절약

- `none` — offload 사용 안 함 (VRAM 충분할 때)
- `model` — 모델 단위 offload
- `sequential` — 가장 강한 offload. VRAM 최소화하지만 느림

<br>

## 프로젝트 구조

```
image-generation/
├── app.py                    # Gradio WebUI 진입점
├── factory.py                # 모델 타입별 파이프라인 생성
├── latent_diffusion.py       # SD 1.5 / 2.0 추론
├── latent_sdxl.py            # SDXL 추론
├── reference_diffusers.py    # diffusers 기반 참조 구현 (Flux 등)
├── text_to_img.py            # 공통 text-to-image 로직
├── components/               # 샘플러 등
└── utils/                    # 시드, 로깅 등 유틸
```

<br>

## 모델 파일

이 레포는 모델 가중치를 포함하지 않는다. 직접 받아서 사용해야 한다.

| 모델 | 추천 출처 |
| --- | --- |
| SDXL | [NoobAI XL](https://civitai.com/models/833294), [Pony Diffusion](https://civitai.com/models/257749) 등 (Civitai) |
| SD 1.5 | `runwayml/stable-diffusion-v1-5` (HuggingFace) |
| SD 2.0 | `stabilityai/stable-diffusion-2-1` |
| Flux | `black-forest-labs/FLUX.1-dev` |

로컬 `.safetensors` 파일은 WebUI에서 드래그앤드롭으로 로드 가능. HuggingFace 모델은 경로 직접 입력으로 사용한다.
