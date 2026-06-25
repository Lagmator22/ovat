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
    """I read a 16 bit mono WAV into float samples the pipeline expects.

    Note to myself: this assumes 16 bit PCM mono audio. If I later need other
    formats I will convert them first. Dividing by 32768 maps the integer
    samples into the range minus one to one, which is what Whisper wants.
    """
    with wave.open(file_path, "rb") as wav:
        frames = wav.readframes(wav.getnframes())
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
    if pipeline is None:
        pipeline = _load_pipeline()
    samples = _read_wav(file_path)
    return str(pipeline.generate(samples))


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
