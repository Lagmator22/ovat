# AI PC verification pass

Paste this whole file to Claude Code on the AI PC (`C:\Users\devcloud\ovat`).
It is ordered so the things that can **change what you say in the meeting**
come first. Stop and report after each numbered block rather than at the end.

Rules for this session:

- Use the venv interpreter: `.\.venv\Scripts\python.exe`, never system python.
- Never push. Commit locally only.
- Every claim you write into a doc must have a command output behind it in this
  session. If you did not run it, say "not verified", do not soften it.
- If something fails, capture the exact error text before changing anything.

---

## 0. Baseline

```bat
git pull
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: green, ~610 tests. Report the exact count.

---

## 1. NPU on OVMS — the claim that was wrong twice

**Background.** OVAT told the mentors "NPU cannot do tool calling" (wrong), then
"OVMS does not start on NPU at all, the export does not compile" (also not the
general truth). OVMS's own demo serves a **tool-calling LLM on NPU** on
LunarLake — the same silicon as this box:
https://github.com/openvinotoolkit/model_server/blob/main/demos/llm_npu/README.md

The documented requirement is a **channel-wise symmetric INT4** export
(`--sym --ratio 1.0 --group-size -1`). The stock `-int4-ov` models are group
quantised, which is why they fail with `[NPU_VCL] Compilation failed
(0x78000004)`.

**Verify all of this. It is the single most important block here.**

1a. Reproduce the failure once more and capture the exact text:

```bat
ovms --pull --source_model OpenVINO/Qwen3.5-0.8B-int4-ov --model_repository_path models --target_device NPU --task text_generation
```

1b. Now pull a model that meets the requirement and serve it:

```bat
ovms --pull --source_model OpenVINO/Qwen3-8B-int4-cw-ov --model_repository_path models --target_device NPU --task text_generation --tool_parser hermes3 --cache_dir .ov_cache --enable_prefix_caching true --max_prompt_len 2000
ovms --add_to_config --config_path models\config.json --model_name OpenVINO/Qwen3-8B-int4-cw-ov --model_path OpenVINO\Qwen3-8B-int4-cw-ov
ovms --rest_port 8000 --config_path models\config.json
```

Check readiness: `curl http://localhost:8000/v1/config`

1c. **The one that matters.** Point an OVAT workflow at that model on NPU and
run an agent that must call a tool (`examples/rag` or `search_docs`). OVMS
documents that on NPU `finish_reason` is **always** `"stop"`, even when it has
decoded a tool call. OVAT was just fixed to dispatch on the tool-call payload
rather than that label (`ovat/agent/loop.py`, and
`tests/test_loop.py::test_a_decoded_tool_call_runs_even_when_finish_reason_says_stop`).

```bat
ovat run <npu-workflow>.yml --input "<something that forces a tool>" --trace npu-trace.json
```

Report: did a tool actually run? What is `finish_reason` in the trace, and what
is `tool_calls`? **If `finish_reason` is `stop` and `tool_calls` is 1+, the fix
is proven on hardware and that is a headline result for the meeting.**

If NPU serving fails for a reason other than the export format, capture it
verbatim — that is the honest finding and it replaces the text in
`docs/ARCHITECTURE.md` (the NPU table) and the README NPU block.

---

## 2. The KV-cache hypothesis — currently unproven

What was told to the mentors: a full KV cache truncates the `<tool_call>` block
mid-generation, so the parser is handed a fragment and correctly refuses it.
Evidence so far is **correlational only**: 4/4 failures at 98-100% cache, 17/17
clean on a fresh server. Nobody has yet shown a full cache **recovering** when
the cache is enlarged, which is what would turn this from a hypothesis into a
mechanism.

Do exactly this, in one session, and keep every trace:

2a. Start OVMS with a deliberately small cache:

```bat
ovat serve <workflow>.yml        # with model.ovms_cache_size_gb set LOW, e.g. 1
```

2b. Drive it until `ovat telemetry` shows the KV cache near 100%: run `ovat
bench`, then several long multi-round agent runs.

2c. With the cache at 98-100%, run a tool-calling prompt with `--trace`.
Capture the trace AND the telemetry numbers **in the same window**.

2d. Without changing anything else, restart with a large cache
(`ovms_cache_size_gb: 8` or whatever fits) and run the **identical** prompt.

Report the pair. Three outcomes and all three are publishable:

- Fails at small cache, succeeds at large → **mechanism confirmed**, say so.
- Fails at both → the cache is not the cause; reopen it as unknown.
- Succeeds at both → not reproducible on demand; keep it as "unreproduced" and
  say that plainly. Do not upgrade a correlation to a cause.

If OVMS reports a truncation or preemption reason in `ovms.log` at the moment
of failure, that log line is the strongest evidence available — grab it.

---

## 3. Telemetry — new code, needs a look on real hardware

The TUI telemetry page now shows a **table of numbers** (now / min / max /
mean) per metric instead of a sparkline per metric, and there is a new `npu`
source that reads the driver's busy counter directly.

3a. Does this Windows build expose an NPU performance counter? This is an open
question the code currently answers with "unknown":

```powershell
Get-Counter -ListSet *NPU*
Get-Counter "\GPU Engine(*)\Utilization Percentage" -MaxSamples 1
```

Report the output of both. If the first prints a counter set, name it — that is
the missing piece for a real Windows NPU number and I will implement it.

3b. `ovat telemetry` and `ovat telemetry --once`. Confirm every source either
shows numbers or states a reason. No silent empty rows.

3c. In the TUI: `/telemetry`. Confirm the Live tab shows per-core CPU rows as
figures, that the numbers update in place without the table jumping, and that
F5 (clear) then repopulates rather than erroring.

3d. Read the **Intel** and **Plano** tabs end to end and tell me whether the
steps are followable by someone who has never used either tool. They were
rewritten from the vendors' own docs; if any step does not match what this
machine actually does, that is a bug in the text.

---

## 4. Engines with no automated live coverage

`tests/test_ovms_live.py` covers only the native loop and LangChain react.
LlamaIndex and openai-agents have been proven by hand once (2026-07-29) and
nothing gates a regression.

4a. Run the live suite: `.\.venv\Scripts\python.exe -m pytest -m live -q`

4b. Run `ovat bench` on a **freshly started** OVMS and confirm all four engines
answer and at least one genuinely calls a tool (look for a citation, not just
an answer).

4c. Then add live tests for the two uncovered engines, modelled exactly on the
two that exist in `tests/test_ovms_live.py`. One "plain chat answers" and one
"actually calls a tool" per engine. Mark them `live` so they auto-skip
elsewhere. Run them; they must pass against the running server.

---

## 5. The README, executed literally

The install path was verified at 1.0.0. The README has changed since, including
today. Nothing checks that its commands still work.

Work through it as a stranger would, in a **scratch directory**, with every
`OVAT_*` variable cleared. Every command in order, including:

- `hf download ...` for each model in the table
- `pip install "ovat[convert]"` then the `optimum-cli export openvino ...`
  command for bge-small — **this one specifically**, it is the only conversion
  step and it is not covered by any test
- `ovat setup`, `ovat init`, `ovat doctor`, `ovat index`, `ovat serve`,
  `ovat run`, `ovat bench`, `ovat chat`, `ovat serve --stop`

Report every command whose output does not match what the README implies. A
wrong flag or a renamed subcommand is exactly what a mentor hits first.

---

## 6. Only if 1-5 are clean

Full TUI pass: `/chat` with `/engine ovms` and a real tool call, `/doctor`,
`/telemetry`, session `/save` and `/load`, indexing progress, themes, palette,
Ctrl-C copy of a selection. Confirm nothing hangs at an 80x24 terminal.

---

## Reporting

For each block: what you ran, what happened, and whether the docs now need to
change. Where a doc claim is now wrong, fix the doc in the same commit as the
finding and reference the command output in the commit body. One logical change
per commit, and every fix ships with a test that fails when the fix is backed
out — verify that, do not assume it.
