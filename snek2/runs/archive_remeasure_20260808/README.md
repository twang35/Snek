# Benchmark and re-measurement results, 2026-08-08

Moved out of `runs/` rather than deleted: **every file here holds real 100-1000 episode
measurements**, and together they are the provenance for the record figure — `b17b-forkseed2`
@1190000 at **4825/5120 = 94.24%** (CI 93.6-94.8), the most heavily measured checkpoint in the
project.

They were moved because `eval_progress.load_runs` groups a "job" by mtime within an hour, so leaving
40 fresh files in `runs/` would have pooled benchmark runs into the next real chart for this arm.

| prefix | what it was |
|---|---|
| `*_bench_w4a/w4b/w10/w10b/w20_p*` | batched-path worker-count benchmark, 4 parallel processes x 800 episodes |
| `*_v2_i4/i5/i10/b5_p*` | the same benchmark after the independent-worker change |
| `champion_b17b_*_bias_indep/_bias_batched` | the paired 1000-episode test that cleared the independent path of bias |
| `champion_b17b_*_hofverify` | the hall-of-fame copy check, 91/100 |
| `champion_b17b_*_smoke_indep` | first end-to-end run of the independent path, 19/20 |

Kept per this project's standing rule that measurements are evidence: `b5c-schlongIS`'s peak became
permanently unmeasurable once its checkpoints rotated out, and that is the mistake this avoids.
