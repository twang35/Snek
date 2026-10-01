"""The `SNEK_R2D2_*` knob names, in a module with no torch so every other algorithm's refusal list can
import them without importing R2D2 (`algos/ppo/algo.py`, `algos/sac/algo.py`, `algos/bbf/algo.py`,
`algos/rainbow/algo.py`)."""

R2D2_KNOBS = ('R2D2_HIDDEN', 'R2D2_STREAM_WIDTH', 'R2D2_SEQ_LENGTH', 'R2D2_BURN_IN', 'R2D2_STRIDE', 'R2D2_PRIORITY_ETA',
              'R2D2_RESCALE', 'R2D2_RESCALE_EPS', 'R2D2_HEAD', 'R2D2_PREV_INPUT', 'R2D2_RECURRENT',
              'R2D2_WINDOWS', 'R2D2_IS_BETA', 'APEX_ALPHA')
