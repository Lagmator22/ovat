# Contributing

Thanks for helping. This page is the short version. `AGENTS.md` has the
full rules and the reasons behind them.

## Set up

```bash
git clone https://github.com/Lagmator22/ovat.git && cd ovat
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

`[dev]` installs every optional engine and the TUI, so the whole suite
can run.

## Run the tests

Always use the venv's own Python, never the system one:

```bash
.venv/bin/python -m pytest -q -n auto            # macOS and Linux
.\.venv\Scripts\python.exe -m pytest -q -n auto  # Windows
```

Tests marked `live` need a running OVMS server and skip by themselves.

## Changes

- Work on a branch made from `main`, one branch per fix or feature, and
  open a pull request. Nothing is pushed to `main` directly.
- Do not rebase or force-push a branch others can see. Fix a mistake with
  a new commit.
- Add files by name (`git add path/to/file`), not `git add .`.
- Every fix comes with a test, and that test must fail without the fix.
  Check it: undo the fix, run the test, see it fail.
- One change per commit. Use a conventional message (`fix(serve): ...`,
  `docs: ...`) whose body says why.
- A new field in `ovat/config/workflow.py` also needs a line in
  `scripts/gen_workflow_reference.py`. Then run that script to regenerate
  `docs/workflow_yaml_reference.md`. A test fails if you forget.
- Pull requests are merged only when every CI job is green.

## Writing

Docs, comments, commit messages and pull requests use plain, simple
English. Please do not use em dashes, or the middle dot as a separator.
