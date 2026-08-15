# Audio + vision: one agent that can listen and look

Ask about a `.wav` and the agent transcribes it, then answers from the
transcript. Ask about an image and it describes it. Both are ordinary tools,
so the model decides when to reach for them.

```
"What is said in sample.wav?"  ─> transcribe    ─> Whisper  ─> text ─┐
"What is in this diagram?"     ─> describe_image ─> VLM     ─> text ─┼─> answer
"What do my notes say?"        ─> search_docs   ─> vectors ─> text ─┘
```

## One model does both halves

`Qwen3.5-4B-int4-ov` is a **unified** export: text generation and image
understanding in the same weights. So the agent's brain and its eyes are one
3.5 GB download, not two.

| | Download |
| --- | --- |
| `OpenVINO/Qwen3.5-4B-int4-ov`, agent **and** vision | 3.5 GB |
| `OpenVINO/whisper-base-int8-ov`, the ears | 0.08 GB |
| **Total** | **~3.6 GB** |

A separate vision model would have been 5 GB on its own; the smallest
dedicated VLM in the OpenVINO org is larger than this entire example.

## Setup

```bash
# The ears. whisper-base is a good accuracy/size trade; whisper-tiny-int8-ov
# (0.05 GB) is faster, whisper-small-int8-ov more accurate.
hf download OpenVINO/whisper-base-int8-ov --local-dir models/whisper-base-int8-ov

# The eyes: the SAME folder as the agent model.
hf download OpenVINO/Qwen3.5-4B-int4-ov --local-dir models/OpenVINO/Qwen3.5-4B-int4-ov
```

**No environment variables.** `workflow.yml` names both models, so the file is
the whole description of this agent:

```yaml
tools:
  - name: transcribe
    type: builtin
    model: models/whisper-base-int8-ov
    device: CPU                 # optional; omit to let the device router pick
  - name: describe_image
    type: builtin
    model: models/OpenVINO/Qwen3.5-4B-int4-ov
```

Point `model` somewhere else to swap either one -- `whisper-tiny-int8-ov` for
speed, any OpenVINO speech-to-text export, any vision-capable export. Nothing
here is Whisper-specific.

Get a sample audio file:

```bash
curl -L https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav \
    -o examples/audio-multimodal/sample.wav
```

## Run it

```bash
ovat serve examples/audio-multimodal/workflow.yml

# listen
ovat run examples/audio-multimodal/workflow.yml \
    -i "What is said in examples/audio-multimodal/sample.wav?"

# look - reusing an image already in this repo, so nothing to download
ovat run examples/audio-multimodal/workflow.yml \
    -i "Describe the image at ovat/assets/intel-phase-1.png"
```

Watch which tool fires:

```bash
ovat run examples/audio-multimodal/workflow.yml \
    -i "What is said in examples/audio-multimodal/sample.wav?" --trace trace.json
```

`trace.json` records each turn, the tool called, its arguments, latency, and
token counts where the server reports them.

## Things worth knowing

- **`transcribe` and `describe_image` are also standalone MCP servers.** Any
  MCP-aware agent can call them, not just OVAT:
  `python -m ovat.tools.transcribe`.
- **Each tool has its own `device:`**, separate from the agent's. Speech-to-text
  defaults to CPU because it is small and CPU latency is fine; the vision model
  follows the GPU recommendation. Set either explicitly to override.
- **The vision model is loaded lazily**, only when `describe_image` is first
  called, so a text-only session never pays for it.
- **Absolute paths are safest** for a tool's `model:`; a relative path is
  resolved from wherever you
  launched the command.
- **`device:` here is about where the TOOL's model runs**, which is a different
  question from the agent's device. No accelerator executes tools: the device
  runs a model, and the loop runs the Python function.
