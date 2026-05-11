"""
Single entry point: build_pipeline() assembles Model + TextEncoder + Sampler + Pipeline.

Usage:
    from factory import build_pipeline
    pipe = build_pipeline("sdxl", "euler_a_native", model_key="...", num_sampling=30, ...)
    img  = pipe.run(null_prompt="...", prompt="...", cfg_guidance=5, target_size=(832, 1216))
"""
from typing import Optional

from components.samplers import get_sampler_class


def build_pipeline(
    model_type: str,
    sampler_name: str,
    model_key: str,
    num_sampling: int = 30,
    device: str = "cuda",
    # SDXL-specific
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
    use_model_vae: bool = False,
    prediction_type: Optional[str] = None,
    timestep_spacing: Optional[str] = None,
    use_compel: bool = False,
    schedule: str = "linear",
    # Flux-specific
    flux_cpu_offload: str = "model",  # "none" | "model" | "sequential"
):
    """
    Assemble and return a ready-to-use Pipeline.

    model_type:     "sd15" | "sd20" | "sdxl" | "flux"
    sampler_name:   "ddim" | "ddim_cfg++" | "euler" | "euler_a" | "euler_a_native"
                    | "euler_cfg++" | "dpm++_2m_cfg++"  (ignored for flux)
    schedule:       "linear" | "karras" | "exponential" | "ays"
                    (applies to euler/dpm samplers; ignored by ddim, euler_a_native, flux)
    """
    if model_type == "flux":
        return _build_flux(model_key, num_sampling, device, flux_cpu_offload)

    sampler_cls = get_sampler_class(sampler_name)

    if model_type == "sdxl":
        return _build_sdxl(sampler_cls, model_key, num_sampling, device,
                           lora_path, lora_scale, use_model_vae,
                           prediction_type, timestep_spacing, use_compel, schedule)
    elif model_type in ("sd15", "sd20"):
        return _build_sd(sampler_cls, model_key, num_sampling, device, use_compel, schedule)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'sd15', 'sd20', 'sdxl', or 'flux'.")


def _build_sdxl(sampler_cls, model_key, num_sampling, device,
                lora_path, lora_scale, use_model_vae,
                prediction_type, timestep_spacing, use_compel, schedule):
    from components.models.sdxl import SDXLModel
    from components.encoders.sdxl import SDXLTextEncoder
    from components.pipelines.sdxl import SDXLPipeline

    model = SDXLModel(
        model_key=model_key,
        num_sampling=num_sampling,
        device=device,
        lora_path=lora_path,
        lora_scale=lora_scale,
        use_model_vae=use_model_vae,
        prediction_type=prediction_type,
        timestep_spacing=timestep_spacing,
    )
    encoder = SDXLTextEncoder(
        tokenizer_1=model.tokenizer_1,
        tokenizer_2=model.tokenizer_2,
        text_enc_1=model.text_enc_1,
        text_enc_2=model.text_enc_2,
        device=device,
        use_compel=use_compel,
    )
    sampler = sampler_cls(model, schedule=schedule)
    return SDXLPipeline(model, encoder, sampler)


def _build_flux(model_key, num_sampling, device, cpu_offload):
    from components.pipelines.flux import FluxPipelineWrapper
    return FluxPipelineWrapper(
        model_key=model_key,
        num_sampling=num_sampling,
        device=device,
        cpu_offload=cpu_offload,
    )


def _build_sd(sampler_cls, model_key, num_sampling, device, use_compel, schedule):
    from components.models.sd import SDModel
    from components.encoders.sd import SDTextEncoder
    from components.pipelines.sd import SDPipeline

    model = SDModel(model_key=model_key, num_sampling=num_sampling, device=device)
    encoder = SDTextEncoder(
        tokenizer=model.tokenizer,
        text_encoder=model.text_encoder,
        device=device,
        use_compel=use_compel,
    )
    sampler = sampler_cls(model, schedule=schedule)
    return SDPipeline(model, encoder, sampler)
