# Snek

This project uses a dedicated conda environment named `snek` (Python 3.10).

Before running any Python scripts or installing packages in this project, activate it:

```
conda activate snek
```

If running commands non-interactively, use `conda run -n snek <command>` instead.

**Caveat:** `conda run` buffers/relays the wrapped process's stdout internally,
even with `python -u`. If you redirect a background process's output to a log
file and poll that file (e.g. while a training run is going), the log can
stay completely empty for 90+ seconds while the process is actually running
fine — and killing it with `kill -9` at that point discards whatever was
sitting in `conda run`'s buffer, permanently. For any background run where you
need to see live/incremental output, invoke the env's python binary directly
instead, e.g.:

```
/opt/miniconda3/envs/snek/bin/python -u snek2.py train > log 2>&1 &
```

Reserve `conda run` for short one-shot commands where only the final
exit/output matters.

## Active development

`snek2/` is the only directory that should be edited going forward. It's a
working copy of `theSchlong/`.

All other directories (`theSchlong/`, `theSchlongCardinalDirs/`, `humanPlayer/`,
etc.) are kept as-is for posterity — do not edit files in them.
