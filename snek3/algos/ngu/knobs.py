"""The `SNEK_NGU_*` knob names, in a module with no torch so every other algorithm's refusal list can import
them without importing NGU (`algos/r2d2/algo.py`, `algos/ppo/algo.py`, `algos/sac/algo.py`, `algos/bbf/algo.py`,
`algos/rainbow/algo.py`). `plans/algoExploration/f-ngu-agent57.md` §1-2."""

NGU_KNOBS = ('NGU_MIXTURES', 'NGU_BETA_MAX', 'NGU_GAMMA_MIN', 'NGU_ASSIGN', 'NGU_RETRACE_LAMBDA', 'NGU_TARGET_EPSILON',
             'NGU_MIN_WINDOWS', 'NGU_EPSILON_PERMUTE', 'NGU_INPUT', 'NGU_EMBED_DIM', 'NGU_EMBED_HIDDEN', 'NGU_EMBED_STEPS',
             'NGU_EMBED_LR', 'NGU_EMBED_L2', 'NGU_RND_HIDDEN', 'NGU_RND_OUT', 'NGU_KNN_K', 'NGU_KERNEL_EPSILON',
             'NGU_CLUSTER_DISTANCE', 'NGU_PSEUDO_COUNT', 'NGU_MAX_SIMILARITY', 'NGU_MEMORY', 'NGU_LIFELONG_MAX')
