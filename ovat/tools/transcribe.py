# ovat/tools/transcribe.py
"""Deliverable 4: the transcribe MCP tool.

Note to myself: this wraps my Whisper speech to text as an MCP tool. An audio
file path goes in, transcript text comes out. I run on CPU first because the
proposal says GPU is an optimization, not a requirement, for this stage.

Same design as search_docs: the real work lives in a plain function so I can
unit test it with a fake pipeline, and the MCP wrapper is a thin layer on top.
"""
import os
import wave

import numpy as np
from fastmcp import FastMCP

# Heavy pipelines, built lazily and cached BY (model, device) rather than in
# one slot. A single global was fine while the model could only come from an
# env var, because then there was only ever one. Now that workflow.yml can name
# a model per tool, two agents in one process -- the bench runs four engines
# back to back -- would silently share whichever pipeline loaded first, and the
# second would transcribe with a model its config never asked for.
_pipelines: dict = {}

#: Fallback when nothing is configured. `models/whisper-base` is the path the
#: docs tell people to export into.
DEFAULT_MODEL_DIR = "models/whisper-base"


def _configured_model(model: str | None = None) -> str:
    """Where the speech-to-text model lives.

    Order: what workflow.yml says, then the env var, then the default. The env
    var stays as a fallback rather than the interface -- it is read HERE and
    not at import time, so setting it late still works, and it is no longer
    the only way to answer the question.

    OVAT_WHISPER_MODEL is kept for the people already using it, but the tool is
    `transcribe`, not `whisper`: any OpenVINO speech-to-text export works, and
    the config field is the name that says so.
    """
    return model or os.environ.get("OVAT_WHISPER_MODEL") or DEFAULT_MODEL_DIR

mcp = FastMCP("transcribe")


def _read_wav(file_path: str):
    """I read a 16 bit mono 16 kHz WAV into float samples the pipeline expects.

    Whisper expects 16 kHz, 16 bit, mono audio. I check those up front and raise
    a clear ValueError instead of silently reading a stereo or 44.1 kHz file,
    which would parse without error but transcribe as garbled or sped-up speech.
    Dividing by 32768 maps the integer samples into minus one to one.
    """
    with wave.open(file_path, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if channels != 1:
        raise ValueError(f"audio must be mono, but this file has {channels} channels")
    if sample_width != 2:
        raise ValueError(f"audio must be 16 bit PCM, but this file is {sample_width * 8} bit")
    if frame_rate != 16000:
        raise ValueError(f"audio must be 16 kHz, but this file is {frame_rate} Hz")
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _resolve_device(device: str | None = None) -> str:
    """The device to load on: config, then env var, then DeviceManager.

    Falls back to CPU if openvino cannot be queried at all, because a tool
    that will not load is worse than a tool on a slower device.
    """
    chosen = device or os.environ.get("OVAT_WHISPER_DEVICE")
    if chosen:
        return chosen
    try:
        from ovat.core.device_manager import DeviceManager
        return DeviceManager().get_whisper_device()
    except Exception:
        return "CPU"


def _load_pipeline(model: str | None = None, device: str | None = None):
    """Build the speech-to-text pipeline once PER (model, device), then reuse.

    Keyed rather than singleton: see _pipelines above for why one slot is not
    enough once the model is configurable.
    """
    key = (_configured_model(model), _resolve_device(device))
    if key not in _pipelines:
        if not os.path.isdir(key[0]):
            # Check the folder BEFORE openvino_genai sees it. A missing path
            # comes back from there as a five-line C++ assertion out of
            # core.cpp, which names an internal .xml the user never chose and
            # says nothing about the setting that is actually wrong. The
            # caller adds the path, so this only has to supply the fix.
            # factory.build_embedder already guards its model this way.
            raise FileNotFoundError(
                "the folder does not exist. Set the transcribe tool's "
                "`model:` in workflow.yml to a local OpenVINO speech-to-text "
                "export, e.g.\n"
                "  hf download OpenVINO/whisper-base-int8-ov "
                "--local-dir models/whisper-base-int8-ov")
        # Imported here, not at the top, so the module loads even on a machine
        # without the model. This is the real OpenVINO GenAI Whisper pipeline.
        import openvino_genai as ov_genai
        _pipelines[key] = ov_genai.WhisperPipeline(*key)
    return _pipelines[key]


def transcribe_impl(file_path: str, language: str = "en", pipeline=None,
                    model: str | None = None, device: str | None = None) -> str:
    """The real logic, kept separate so my tests can pass a fake pipeline.

    Note to myself: I check the file exists first and return a clear error
    string instead of letting a missing path blow up the whole agent.
    """
    if not os.path.isfile(file_path):
        return f"Error: I could not find an audio file at: {file_path}"
    try:
        samples = _read_wav(file_path)
    except Exception as exc:
        return (f"Error: {exc}. Convert it to 16 kHz, 16 bit, mono first.")
    if pipeline is None:
        try:
            pipeline = _load_pipeline(model, device)
        except Exception as exc:
            # Name the path that was tried. "Error loading model: ..." with no
            # path left the reader guessing whether the config was read at all.
            return (f"Error loading the transcribe model from "
                    f"{_configured_model(model)!r}: {exc}")
    try:
        # Strip any wrapper the caller already added. The SCHEMA asks for a
        # bare code ("en"), but this argument comes from a MODEL, and a model
        # that has seen Whisper prompts writes the full token "<|en|>" often
        # enough to matter. Wrapping that again gives "<|<|en|>|>", which the
        # tokenizer cannot match, so it silently transcribes as the wrong
        # language instead of failing.
        code = language.strip().strip("<|>") or "en"
        return str(pipeline.generate(samples, language=f"<|{code}|>"))
    except Exception as exc:
        return f"Error transcribing audio: {exc}"


# The OpenAI-style tool schema my agent loop shows the model. Co-located with
# the tool so the model's description and the real function stay in sync.
SCHEMA = {
    "type": "function",
    "function": {
        "name": "transcribe",
        "description": "Transcribe a spoken audio file into text. Use when the "
                       "user gives a path to an audio recording and wants the words.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "path to a WAV audio file"},
                # "default" keeps this schema the complete contract; the
                # LangChain argument model is derived from it.
                "language": {"type": "string", "description": "language code, e.g. en",
                             "default": "en"},
            },
            "required": ["file_path"],
        },
    },
}


@mcp.tool
def transcribe(file_path: str, language: str = "en") -> str:
    """Transcribe a spoken audio file into text.

    Use me when the user gives a path to an audio recording and wants the
    words in it. I take a WAV file path and return the transcript as text.
    """
    return transcribe_impl(file_path, language)


if __name__ == "__main__":
    # Note to myself: runs transcribe as a standalone MCP server.
    # My workflow YAML launches me with: python transcribe.py
    mcp.run()
