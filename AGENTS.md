# Project Instructions

- Ask for permission before making a modification outside the current scope, because the user may have changed previously written code without notice.

## Pre-authorized temporary workspace

- This section is an explicit exception to the general approval rule above.
- For every task started from this project, the user gives standing authorization to create files and directories and to modify, overwrite, or replace existing files anywhere under `/private/tmp/`.
- Treat these filesystem operations as already approved. Do not ask for additional confirmation merely because the target is under `/private/tmp/`.
- This authorization does not extend to deleting unrelated pre-existing files or causing side effects outside `/private/tmp/`.

## Notebook validation

- In the managed local environment, treat Jupyter/IPython kernel failures caused by ZeroMQ or local socket binding errors (for example, `Operation not permitted`) as an environment restriction, not as evidence of a notebook code failure.
- Once that error signature is confirmed, do not repeatedly retry starting a Jupyter kernel, including in approved or escalated mode. Switch directly to a socket-free smoke test that executes notebook cells sequentially in one process and preserves shared state, working directory, cell order, and fail-fast behavior.
- Prefer an in-process IPython runner when the notebook contains magics, shell escapes, rich display calls, or other IPython-only syntax. A plain exported Python script is acceptable only when all cells are valid Python.
- The fallback smoke test must cover the notebook's complete intended path (for example: setup → data → training → backtest → figures). Report clearly that this validates execution without a real Jupyter kernel and note any notebook behaviors the fallback cannot faithfully reproduce.

## Python and notebook environment

- Use `/opt/anaconda3/envs/finrl_exp/bin/python` for Python commands and package inspection in this project; the default shell Python belongs to the older Anaconda base environment.
- Run scripts with `/opt/anaconda3/envs/finrl_exp/bin/python path/to/script.py`, or `conda run -n finrl_exp python path/to/script.py`.
- For notebooks, select the `finrl_exp` kernel (display name: `Python (finrl_exp)`). Preserve existing notebook kernel metadata unless the task explicitly requires changing it.
- Run notebook cells from the notebook's directory unless its documented workflow specifies another working directory.
- Normal project-local Python and notebook execution is allowed within the active sandbox. Request user approval for operations requiring additional permissions; do not add blanket Python command approval rules.
- Package installation into `/opt/anaconda3/envs/finrl_exp` requires approval when that directory is outside the writable roots. Use this environment's `python -m pip`, not base `pip`.
