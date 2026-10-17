# ovat/core/precision.py
"""Keep an OpenVINO CPU pipeline in f32 where its default would be f16.

One place for the rule, used by every local pipeline OVAT builds (vision,
speech). Needs openvino, which is imported only when a pipeline is built.
"""


def precision_properties(device: str, core=None) -> dict:
    """Pipeline properties that keep a CPU run in f32 where it would not be.

    On an ARM CPU (Apple Silicon) OpenVINO's CPU plugin defaults
    INFERENCE_PRECISION_HINT to f16. Qwen2-VL overflows in f16 and, a few
    tokens in, every logit is NaN, so greedy picks token 0 forever -- which in
    Qwen's vocabulary is "!". Measured on an M-series Mac: "The image features
    the logo!!!!!!..." at the default, and "The image features the logo of
    OpenVINO, which is a toolkit for AI models." with f32 (about 2.8x slower,
    58.6 s vs 21.2 s for 60 tokens, but the fast answer was garbage).

    Whisper has the same problem: on an M-series Mac, 14 s of fast speech
    through WhisperPipeline came back as "Today!!!!!!..." in 3.2 s at the
    default, and as the correct sentence in 0.87 s with f32 (whisper-base,
    OVAT's voice dictation and the transcribe tool).

    Only when the CPU's own default is f16, so an x86 CPU (f32 already, or
    bf16 on hardware that has it) keeps exactly what it had: those were never
    measured broken. GPU and NPU are left alone for the same reason.
    """
    import openvino as ov
    if (device or "").upper() != "CPU":
        return {}
    try:
        core = core or ov.Core()
        default = core.get_property("CPU", "INFERENCE_PRECISION_HINT")
    except Exception:                  # a property it cannot read: change nothing
        return {}
    if default == ov.Type.f16:
        return {"INFERENCE_PRECISION_HINT": "f32"}
    return {}
