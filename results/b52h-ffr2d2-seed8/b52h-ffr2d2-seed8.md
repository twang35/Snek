# b52h-ffr2d2-seed8

step **1,300,000** · 1300 evals · trailing **94.66** · peak **94.79** @1,042,000 · sef **76.7** · best30 **97.6** @1,237,000

## Config

| | |
|---|---|
| adam_epsilon | 0.001 |
| algo | r2d2 |
| apex_alpha | 7.0 |
| batch_size | 64 |
| collect_envs | 32 |
| discount | 0.997 |
| dist_atoms | 51 |
| dist_v_max | 110.0 |
| dist_v_min | -10.0 |
| epsilon_anneal_steps | 0 |
| epsilon_schedule | apex |
| eval_interval | 1000 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| gradient_clipping | 40.0 |
| graph_eval_episodes | 100 |
| init_from | None |
| initial_collect_steps | 20000 |
| initial_epsilon | 0.4 |
| learning_rate | 0.0001 |
| max_steps | 1300000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.0 |
| n_step_update | 5 |
| priority_exponent | 0.9 |
| r2d2_burn_in | 40 |
| r2d2_head | scalar |
| r2d2_hidden | 512 |
| r2d2_is_beta | 0.6 |
| r2d2_prev_input | True |
| r2d2_priority_eta | 0.9 |
| r2d2_recurrent | dense |
| r2d2_rescale | True |
| r2d2_rescale_eps | 0.001 |
| r2d2_seq_length | 120 |
| r2d2_stream_width | 512 |
| r2d2_stride | 40 |
| r2d2_windows | 30000 |
| replay_ratio | 0.003125 |
| seed | 8 |
| target_update_period | 2500 |
| torch_threads | 1 |

![b52h-ffr2d2-seed8](b52h-ffr2d2-seed8.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.17 | 0.17 | 0.0 | 2.0 | -0.379 | 0.0 | 0.4 |
| 2000 | 2.39 | 1.28 | 0.0 | 13.0 | 1.538 | 0.0 | 0.4 |
| 3000 | 6.84 | 3.13 | 0.0 | 28.0 | 5.479 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1289000 | 94.03 | 94.63 | 12.0 | 95.0 | 186.545 | 94.0 | 0.4 |
| 1290000 | 94.69 | 94.65 | 76.0 | 95.0 | 189.281 | 96.0 | 0.4 |
| 1291000 | 94.96 | 94.65 | 92.0 | 95.0 | 191.536 | 98.0 | 0.4 |
| 1292000 | 94.72 | 94.65 | 83.0 | 95.0 | 188.291 | 95.0 | 0.4 |
| 1293000 | 94.99 | 94.66 | 94.0 | 95.0 | 192.516 | 99.0 | 0.4 |
| 1294000 | 94.96 | 94.68 | 92.0 | 95.0 | 191.493 | 98.0 | 0.4 |
| 1295000 | 94.73 | 94.67 | 80.0 | 95.0 | 188.357 | 95.0 | 0.4 |
| 1296000 | 93.85 | 94.64 | 13.0 | 95.0 | 186.371 | 94.0 | 0.4 |
| 1297000 | 94.9 | 94.65 | 85.0 | 95.0 | 192.521 | 99.0 | 0.4 |
| 1298000 | 94.5 | 94.66 | 70.0 | 95.0 | 187.02 | 94.0 | 0.4 |
| 1299000 | 94.79 | 94.66 | 81.0 | 95.0 | 190.39 | 97.0 | 0.4 |
| 1300000 | 94.68 | 94.66 | 82.0 | 95.0 | 189.263 | 96.0 | 0.4 |
