# CLAUDE.md

## Python environment

All Python in this repo runs in the `irl` conda environment. Nothing else has the
dependencies — the conda base env has no `torch` at all, so a bare `python` will fail.

Use the interpreter path directly:

```
/opt/anaconda3/envs/irl/bin/python script.py
```

Same for any env-provided tool (`pip`, `jupyter`, `pytest`): call it from
`/opt/anaconda3/envs/irl/bin/`.

Do not use `conda activate irl` in a tool-invoked shell. It needs an initialized
interactive shell and silently leaves you in base when it fails. `conda run -n irl python`
works but buffers output, so prefer the direct path.

Environment: Python 3.14.4, torch 2.11.0, numpy 2.4.4.

## Notes

- `README.md` is inherited from the upstream paper repo (Tran et al., 2022). Its setup
  section (python3.6/3.7, `pip3 install .`, Docker) and the `Makefile` docker targets
  describe the original project, not this workflow. Use this file for how to run things.
