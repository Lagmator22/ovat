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

# I build the heavy Whisper pipeline lazily and cache it here, so importing
# this module stays cheap and my tests do not need the model on disk.
_pipeline = None

# Where my converted Whisper model lives. I can override it with an env var.
WHISPER_MODEL_DIR = os.environ.get("OVAT_WHISPER_MODEL", "models/whisper-base")

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


def _load_pipeline():
    """I build the Whisper pipeline once and reuse it after that."""
    global _pipeline
    if _pipeline is None:
        # I import here, not at the top, so the module loads even on a machine
        # without the model. This is the real OpenVINO GenAI Whisper pipeline.
        import openvino_genai as ov_genai
        _pipeline = ov_genai.WhisperPipeline(WHISPER_MODEL_DIR, "CPU")
    return _pipeline


def transcribe_impl(file_path: str, language: str = "en", pipeline=None) -> str:
    """The real logic, kept separate so my tests can pass a fake pipeline.

    Note to myself: I check the file exists first and return a clear error
    string instead of letting a missing path blow up the whole agent.
    """
    if not os.path.isfile(file_path):
        return f"Error: I could not find an audio file at: {file_path}"
    try:
        samples = _read_wav(file_path)
    except ValueError as exc:
        return f"Error: {exc}. Convert it to 16 kHz, 16 bit, mono first."
    if pipeline is None:
        pipeline = _load_pipeline()
    # Pass the requested language in Whisper's token form (en -> <|en|>) so the
    # model transcribes that language instead of guessing it.
    return str(pipeline.generate(samples, language=f"<|{language}|>"))


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
                "language": {"type": "string", "description": "language code, e.g. en"},
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
    return transcribe_impl(file_path, language, _pipeline)


if __name__ == "__main__":
    # Note to myself: runs transcribe as a standalone MCP server.
    # My workflow YAML launches me with: python transcribe.py
    mcp.run()
