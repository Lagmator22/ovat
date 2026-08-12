# AI PC finalisation pass

Paste this whole file to Claude Code on the AI PC (`C:\Users\devcloud\ovat`).

Goal: end this session able to say "OVAT is finished and everything in the
docs has been run on hardware", then record the demo. Ordered so the blocks
that could **change what is said in the mentor meeting** come first.

Ignore A2A and every other stretch goal. They are declared out of scope in
`docs/ARCHITECTURE.md` and that is the correct answer, not a gap.

## Rules for this session

- Use the venv interpreter: `.\.venv\Scripts\python.exe`, never system python.
- **Research before implementing.** Web-search the OVMS / OpenVINO / Textual /
  huggingface docs and their GitHub repos before writing code against them or
  writing a claim into a doc. Several bugs in this repo came from a guess that
  looked reasonable. If you cannot find a citation, say the claim is unverified
  rather than softening it.
- Every claim written into a doc must have command output behind it from **this
  session**. If you did not run it, write "not verified".
- Every fix ships with a test that **fails with the fix backed out**. Verify
  that; do not assume it.
- One logical change per commit. Conventional message, body explains WHY.
- Do not add `Co-Authored-By` trailers.
- Commit locally. Ask before pushing.

---

## 0. Baseline

```bat
git pull
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: **624 passed, some skipped**. Report the exact numbers. The `live`
and `rag` markers auto-skip until a server and models are present.

Six commits landed today. Read them first, they set up most of what follows:

```bat
git log --oneline -6
```

The important one is `fix(agent): dispatch on the tool call, not on
finish_reason`. Block 2 is what proves it on hardware.

---

## 1. Serve, and the four engines

```bat
ovat setup
ovat doctor
ovat serve examples/document-qa.yml
ovat run examples/document-qa.yml --input "<question that forces a tool>" --trace t.json
ovat bench examples/document-qa.yml --input "<same>"
```

Confirm from `t.json`, not from the prose answer: `tool_calls` is 1 or more,
`undecoded_tool_call` is false, `failed` is false. A fluent answer with
`tool_calls: 0` is the silent failure this project has been bitten by twice.

`ovat bench` must produce the four-engine table with all four answering.

---

## 2. NPU — the claim that was wrong twice

**Read this before running anything.** OVAT told the mentors "NPU cannot do
tool calling" (wrong), then "OVMS does not start on NPU at all, the export does
not compile" (not the general truth either). OVMS supports NPU text generation,
demonstrates it **with `--tool_parser hermes3`**, and states it was tested on
**LunarLake — this machine's silicon**:
https://github.com/openvinotoolkit/model_server/blob/main/demos/llm_npu/README.md

The requirement is a **channel-wise symmetric INT4** export
(`--sym --ratio 1.0 --group-size -1`). The stock `-int4-ov` models are group
quantised, which is why they fail with `[NPU_VCL] Compilation failed
(0x78000004)`. OpenVINO publishes an `-int4-cw-ov` family for exactly this.

2a. Reproduce the failure once and capture the exact text:

```bat
ovms --pull --source_model OpenVINO/Qwen3.5-0.8B-int4-ov --model_repository_path models --target_device NPU --task text_generation
```

2b. Pull a model that meets the requirement and serve it:

```bat
ovms --pull --source_model OpenVINO/Qwen3-8B-int4-cw-ov --model_repository_path models --target_device NPU --task text_generation --tool_parser hermes3 --cache_dir .ov_cache --enable_prefix_caching true --max_prompt_len 2000
ovms --add_to_config --config_path models\config.json --model_name OpenVINO/Qwen3-8B-int4-cw-ov --model_path OpenVINO\Qwen3-8B-int4-cw-ov
ovms --rest_port 8000 --config_path models\config.json
curl http://localhost:8000/v1/config
```

2c. **The block that matters.** Point an OVAT workflow at that model on NPU and
run an agent that must call a tool.

OVMS documents that on NPU `finish_reason` is **always** `"stop"`, even when a
tool call was decoded. OVAT used to branch on that label and would have read
the call as prose — answering fluently while running no tool. It now dispatches
on the payload (`ovat/agent/loop.py`;
`tests/test_loop.py::test_a_decoded_tool_call_runs_even_when_finish_reason_says_stop`).

```bat
ovat run <npu-workflow>.yml --input "<forces a tool>" --trace npu-trace.json
```

Report `finish_reason` and `tool_calls` from the trace. **If `finish_reason` is
`stop` and `tool_calls` is 1+, the fix is proven on hardware — that is the
headline result for the meeting.**

If NPU serving fails for a reason *other* than export format, capture it
verbatim; it replaces the NPU table in `docs/ARCHITECTURE.md` and the README
NPU block. Do not soften a failure into a success.

---

## 3. KV cache — a hypothesis that is one experiment from being a mechanism

Told to the mentors: a full KV cache truncates the `<tool_call>` block
mid-generation, so the parser is handed a fragment and correctly refuses it.
Evidence is **correlational only** — 4/4 failures at 98-100%, 17/17 clean on a
fresh server. Nobody has shown a full cache **recovering** when enlarged.

OVAT can now read the cache figure itself. `OVMSLogSource` parses OVMS's own
log line (`type: dynamic, cache usage: 98.5% of 3.60 GB`), because OVMS
documents that text-generation metrics are *not* on its `/metrics` endpoint.
`ovat run --telemetry` records it beside the trace, and `ovat run` warns above
95% before calling the model.

3a. Serve with a deliberately small cache (`model.ovms_cache_size_gb: 1`).
`--cache_size` is in GB; that is verified against OVMS's docs.

3b. Drive it to near-100%: `ovat bench`, then several long multi-round runs.
Watch it climb with `ovat telemetry`.

3c. At 98-100%, run a tool-calling prompt:

```bat
ovat run <workflow>.yml --input "<forces a tool>" --trace full.json --telemetry full.jsonl
```

Confirm the pre-run warning fires. Keep both files.

3d. Restart with a large cache (`ovms_cache_size_gb: 8`), change **nothing
else**, run the identical prompt.

Report the pair. All three outcomes are publishable:

- Fails small, succeeds large → **mechanism confirmed.** Say so plainly.
- Fails both → the cache is not the cause. Reopen it as unknown.
- Succeeds both → not reproducible on demand. Keep it as "unreproduced".

Do not upgrade a correlation to a cause. If `ovms.log` names a truncation or
preemption reason at the moment of failure, that line is the best evidence
available — grab it.

---

## 4. Telemetry — new code, first run on real hardware

The Live tab is now a **table of figures** (source, metric, now, min, max,
mean), not a sparkline per metric. Verified on macOS: per-core rows
`cpu0_pct`..`cpu9_pct` render as numbers and update in place. On this machine
the `npu` and `ovms` rows should populate too.

4a. Does this Windows build expose an NPU performance counter? **This is an
open question the code currently answers with "unknown", and your output
decides whether I implement it:**

```powershell
Get-Counter -ListSet *NPU*
Get-Counter "\GPU Engine(*)\Utilization Percentage" -MaxSamples 1
```

The GPU one is documented and should work. If the first prints a counter set,
name it exactly — that is the missing piece for a real Windows NPU number.
(`NPUSource` reads the driver's `npu_busy_time_us` on Linux; Windows exposes
NPU usage via DXCore rather than a documented counter path, so nothing was
guessed there.)

4b. `ovat telemetry`, then `ovat telemetry --once`. Every source must either
show numbers or state a reason. No silent empty rows.

4c. In the TUI: `/telemetry`. Confirm per-core CPU rows show as figures, the
numbers update without the table jumping or resetting scroll, F5 clears and
repopulates without error, and the KV cache row appears while a server runs.

4d. Read the **Intel** and **Plano** tabs end to end. They were rewritten from
the vendors' own docs (plano's supported-platform list and full `trace` command
set; Intel UT's `ut-vars.cmd` setup). Follow them literally. Any step that does
not match what this machine does is a bug in the text — fix the text.

Note plano ships **no Windows binary**; `planoai up` needs `--docker` or WSL2
here. That is documented in the tab.

---

## 5. Live tests for the two uncovered engines

`tests/test_ovms_live.py` covers only the native loop and LangChain react.
LlamaIndex and openai-agents were proven by hand once (2026-07-29) and nothing
gates a regression.

```bat
.\.venv\Scripts\python.exe -m pytest -m live -q
```

Then add, modelled exactly on the two that exist: one "plain chat answers" and
one "actually calls a tool" per engine, marked `live` so they auto-skip
elsewhere. Assert on a **citation or a tool_calls count**, never on the answer
reading plausibly. Run them against the live server; they must pass.

---

## 6. The README, executed literally

The install path was verified at 1.0.0. The README has changed since, twice
today. Nothing tests that its commands still work.

In a **scratch directory**, with every `OVAT_*` variable cleared, as a stranger
would, run every command in order:

- `pip install ovat`, then `hf download ...` for each model in the table.
  (`hf` comes from `huggingface-hub`, now floored at `>=0.34` because that is
  where `hf` replaced `huggingface-cli`. Confirm `hf` is on PATH from a clean
  install — that floor was added today and has not been installed fresh.)
- `pip install "ovat[convert]"` then the `optimum-cli export openvino ...`
  command for bge-small. **This one specifically** — it is the only conversion
  step and no test covers it.
- `ovat setup`, `init`, `doctor`, `index`, `serve`, `run`, `bench`, `chat`,
  `serve --stop`.

Report every command whose output does not match what the README implies.

---

## 7. Full TUI pass

```bat
ovat tui
```

- `/chat`, then `/engine ovms` — confirm a **real tool call** with a citation
- streaming, foldable reasoning, multi-line input, paste, Up/Down history
- `/save` and `/load`, session picker
- `/doctor` — DataTable, severity sort, per-row copy
- `/telemetry` — as block 4c
- `/index` with the progress bar
- themes, command palette, turn separators
- Ctrl-C copies a selection
- **Resize to 80x24 and confirm nothing hangs.** `#masthead` height is 17 for
  this reason; 18 hangs. Do not change it.

---

## 8. Close it out

Only once 0-7 are clean:

1. Update `docs/ARCHITECTURE.md`'s layer status matrix with what is now
   verified on hardware, and the "outstanding" line with what genuinely remains.
2. Update `AGENTS.md` — its "Still open" list is stale (it lists OpenTelemetry
   as open; Layer 7 shipped) and its landmines should gain the NPU export
   requirement and the `finish_reason` behaviour.
3. Bump the version, tag, and report what changed. Do not publish to PyPI
   without asking.

Report per block: what you ran, what happened, what the docs now say. Where a
doc claim is wrong, fix it in the same commit as the finding and quote the
command output in the commit body.
