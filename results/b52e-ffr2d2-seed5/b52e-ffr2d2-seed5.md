# b52e-ffr2d2-seed5

step **1,300,000** · 1300 evals · trailing **94.81** · peak **94.85** @1,188,000 · sef **76.2** · best30 **97.9** @1,186,000

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
| seed | 5 |
| target_update_period | 2500 |
| torch_threads | 1 |

![b52e-ffr2d2-seed5](b52e-ffr2d2-seed5.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.15 | 0.15 | 0.0 | 1.0 | -0.399 | 0.0 | 0.4 |
| 2000 | 2.79 | 1.47 | 0.0 | 9.0 | 0.79 | 0.0 | 0.4 |
| 3000 | 4.03 | 2.32 | 0.0 | 17.0 | 1.985 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1289000 | 94.95 | 94.72 | 92.0 | 95.0 | 191.535 | 98.0 | 0.4 |
| 1290000 | 94.96 | 94.72 | 92.0 | 95.0 | 191.511 | 98.0 | 0.4 |
| 1291000 | 94.86 | 94.72 | 88.0 | 95.0 | 190.369 | 97.0 | 0.4 |
| 1292000 | 94.96 | 94.74 | 91.0 | 95.0 | 192.565 | 99.0 | 0.4 |
| 1293000 | 94.97 | 94.74 | 93.0 | 95.0 | 191.481 | 98.0 | 0.4 |
| 1294000 | 94.97 | 94.74 | 93.0 | 95.0 | 191.464 | 98.0 | 0.4 |
| 1295000 | 94.87 | 94.75 | 91.0 | 95.0 | 189.352 | 96.0 | 0.4 |
| 1296000 | 94.34 | 94.74 | 64.0 | 95.0 | 188.816 | 96.0 | 0.4 |
| 1297000 | 94.8 | 94.74 | 80.0 | 95.0 | 190.304 | 97.0 | 0.4 |
| 1298000 | 94.85 | 94.81 | 83.0 | 95.0 | 191.461 | 98.0 | 0.4 |
| 1299000 | 94.85 | 94.81 | 84.0 | 95.0 | 190.372 | 97.0 | 0.4 |
| 1300000 | 94.89 | 94.81 | 88.0 | 95.0 | 191.474 | 98.0 | 0.4 |
