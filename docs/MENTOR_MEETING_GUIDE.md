# OVAT Mentor Meeting Guide: Talking Points & Demo Script

> **Purpose**: This guide equips you for your mentor sync with **Ravi Panchumarthy** and **Freddy Chiu**. It summarizes key accomplishments, design decisions, answers to mentor spike questions, and a step-by-step live demo script.

---

## 🎯 Executive Summary (Your 60-Second Elevators Pitch)

> *"OVAT (OpenVINO Agentic Toolkit) achieves our core mission: **'One YAML + One Command'** to run hardware-accelerated tool-calling agents on Intel AI PCs and local dev machines.*
> 
> *We have completed all 4 framework engines (Native Loop, LangChain, LlamaIndex, OpenAI Agents SDK), side-by-side benchmarking (`ovat bench`), built-in RAG and Whisper/Vision tools, a Claude-Code style TUI, and zero-code OpenTelemetry observability fronted by Plano gateway on OVMS."*

---

## 💡 Key Discussion Points & Answers to Mentor Spikes

### 1. Ravi's Spike Question: *"Does Plano's `/v1` default path break OVMS `/v3`?"*
- **Answer**: **No, it works seamlessly!** 
- In `plano-config.yaml`, we configure `base_url: http://<BRIDGE_IP>:8001/v3`. Plano automatically parses `/v3` out of `base_url` and uses it as the upstream path prefix when calling OVMS.
- To resolve schema matching between Plano's WASM filter (which expects a top-level `"id"` string field in JSON responses) and OVMS, we created a lightweight 30-line bridge (`examples/plano/ovms_id_bridge.py`). It handles Plano's chunked HTTP requests and injects the `"id"` field.

### 2. Telemetry & Observability (GSoC Milestone)
- **Zero-Code OTEL Tracing**: By fronting OVMS with Plano, every request generates real-time OpenTelemetry trace spans in `planoai obs` (latency, TTFT, token counts, status codes).
- **Zero Code Bloat in OVAT**: We achieved enterprise OpenTelemetry telemetry without adding 3,900+ lines of OTEL dependencies inside OVAT itself (unlike NeMo).

### 3. Multi-Engine Parity & Strict Architecture
- **Single Source of Truth**: Every tool defines its contract in a co-located `SCHEMA` dict. All 4 engines (Native, LangChain, LlamaIndex, OpenAI Agents) derive their schema models dynamically via `ovat/agent/arg_models.py`.
- **Accurate Telemetry & Peak RSS**: Memory usage (`Peak RSS`) is actively sampled on a background thread during runs because post-run single readings miss memory allocations freed by Python GC.

---

## 🎬 Live Demo Script (Step-by-Step)

Follow this sequence during your mentor presentation:

### Step 1: Demonstrate Native Tool-Calling & Reasoning
Run a native agent with tool calls on OVMS / GenAI:
```bash
ovat run examples/workflow.yml --input "what tools do you have available?"
```
*Point out: The native loop handles Hermes-3 / Qwen-3 tool calling, parses reasoning tags, and records per-turn token usage.*

### Step 2: Demonstrate 4-Engine Benchmarking (`ovat bench`)
Run side-by-side benchmark comparison:
```bash
ovat bench examples/workflow.yml --input "what tools do you have available?"
```
*Point out: All 4 engines run against the exact same workflow YAML and produce a comparative Rich table showing execution time, peak memory RSS, and status.*

### Step 3: Demonstrate Plano Gateway & OpenTelemetry Dashboard
1. On Windows `cmd.exe`:
   ```cmd
   python examples\plano\ovms_id_bridge.py
   ```
2. In WSL2 Terminal 1:
   ```bash
   planoai obs
   ```
3. In WSL2 Terminal 2:
   ```bash
   ovat run examples/plano/workflow.yml --input "what tools do you have available?"
   ```
*Point out: The live `planoai obs` dashboard immediately lights up green with `200 OK`, showing TTFT (Time-To-First-Token) and latency metrics!*

---

## ❓ Anticipated Mentor Questions & Answers

| Question | Recommended Answer |
| :--- | :--- |
| **"Why do we need `ovms_id_bridge.py`?"** | *"Plano's WASM filter strictly requires a top-level `"id"` string field in JSON responses. OVMS returns valid OpenAI JSON but omits `"id"`. The bridge acts as a clean 30-line adapter so neither OVMS binary nor Plano WASM source needs modification."* |
| **"How does OVAT support macOS if OVMS only runs on x86?"** | *"OVAT features a dual serving architecture: local dev machines (like macOS) use `openvino_genai` natively on CPU for `ovat chat` and TUI, while AI PCs use OVMS over HTTP on Arc GPU/NPU."* |
| **"Is the CLI decoupled from the TUI?"** | *"Yes! `[tui]` is an optional extra. The core CLI has zero import dependency on `textual` or `pyfiglet`, enforced by automated isolation tests (`test_tui_isolation.py`)."* |
