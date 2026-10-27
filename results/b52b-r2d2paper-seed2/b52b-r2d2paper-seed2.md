# b52b-r2d2paper-seed2

step **1,300,000** · 1300 evals · trailing **94.13** · peak **94.75** @936,000 · sef **55.8** · best30 **97.1** @1,268,000

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
| r2d2_recurrent | lstm |
| r2d2_rescale | True |
| r2d2_rescale_eps | 0.001 |
| r2d2_seq_length | 120 |
| r2d2_stream_width | 512 |
| r2d2_stride | 40 |
| r2d2_windows | 30000 |
| replay_ratio | 0.003125 |
| seed | 2 |
| target_update_period | 2500 |
| torch_threads | 1 |

![b52b-r2d2paper-seed2](b52b-r2d2paper-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.37 | 0.37 | 0.0 | 3.0 | -1.919 | 0.0 | 0.4 |
| 2000 | 1.95 | 1.16 | 1.0 | 3.0 | -2.535 | 0.0 | 0.4 |
| 3000 | 2.04 | 1.45 | 0.0 | 5.0 | -1.161 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1289000 | 93.79 | 94.26 | 0.0 | 95.0 | 186.436 | 94.0 | 0.4 |
| 1290000 | 92.08 | 94.16 | 0.0 | 95.0 | 185.761 | 95.0 | 0.4 |
| 1291000 | 94.65 | 94.16 | 84.0 | 95.0 | 187.181 | 94.0 | 0.4 |
| 1292000 | 94.85 | 94.19 | 88.0 | 95.0 | 190.432 | 97.0 | 0.4 |
| 1293000 | 94.04 | 94.22 | 0.0 | 95.0 | 190.675 | 98.0 | 0.4 |
| 1294000 | 93.77 | 94.19 | 0.0 | 95.0 | 186.327 | 94.0 | 0.4 |
| 1295000 | 93.66 | 94.14 | 0.0 | 95.0 | 186.239 | 94.0 | 0.4 |
| 1296000 | 94.68 | 94.14 | 76.0 | 95.0 | 190.251 | 97.0 | 0.4 |
| 1297000 | 94.72 | 94.14 | 82.0 | 95.0 | 189.332 | 96.0 | 0.4 |
| 1298000 | 94.84 | 94.14 | 88.0 | 95.0 | 190.478 | 97.0 | 0.4 |
| 1299000 | 94.87 | 94.13 | 90.0 | 95.0 | 190.458 | 97.0 | 0.4 |
| 1300000 | 93.96 | 94.13 | 0.0 | 95.0 | 189.542 | 97.0 | 0.4 |
