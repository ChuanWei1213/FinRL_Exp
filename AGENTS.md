# Project Instructions

- Ask for permission before making a modification outside the current scope, because the user may have changed previously written code without notice.

## Notebook validation

- In the managed local environment, treat Jupyter/IPython kernel failures caused by ZeroMQ or local socket binding errors (for example, `Operation not permitted`) as an environment restriction, not as evidence of a notebook code failure.
- Once that error signature is confirmed, do not repeatedly retry starting a Jupyter kernel, including in approved or escalated mode. Switch directly to a socket-free smoke test that executes notebook cells sequentially in one process and preserves shared state, working directory, cell order, and fail-fast behavior.
- Prefer an in-process IPython runner when the notebook contains magics, shell escapes, rich display calls, or other IPython-only syntax. A plain exported Python script is acceptable only when all cells are valid Python.
- The fallback smoke test must cover the notebook's complete intended path (for example: setup → data → training → backtest → figures). Report clearly that this validates execution without a real Jupyter kernel and note any notebook behaviors the fallback cannot faithfully reproduce.
