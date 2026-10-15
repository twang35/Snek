# b52f-ffr2d2-seed6

step **1,300,000** · 1300 evals · trailing **94.65** · peak **94.82** @1,202,000 · sef **74.7** · best30 **97.5** @944,000

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
| seed | 6 |
| target_update_period | 2500 |
| torch_threads | 1 |

![b52f-ffr2d2-seed6](b52f-ffr2d2-seed6.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.12 | 1.12 | 1.0 | 3.0 | 0.166 | 0.0 | 0.4 |
| 2000 | 3.0 | 2.06 | 0.0 | 9.0 | -0.141 | 0.0 | 0.4 |
| 3000 | 8.34 | 4.15 | 0.0 | 26.0 | 4.92 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1289000 | 94.33 | 94.65 | 74.0 | 95.0 | 185.677 | 93.0 | 0.4 |
| 1290000 | 94.57 | 94.64 | 52.0 | 95.0 | 192.156 | 99.0 | 0.4 |
| 1291000 | 94.68 | 94.63 | 72.0 | 95.0 | 191.294 | 98.0 | 0.4 |
| 1292000 | 94.4 | 94.62 | 52.0 | 95.0 | 190.894 | 98.0 | 0.4 |
| 1293000 | 95.0 | 94.63 | 95.0 | 95.0 | 193.566 | 100.0 | 0.4 |
| 1294000 | 94.96 | 94.64 | 92.0 | 95.0 | 191.48 | 98.0 | 0.4 |
| 1295000 | 94.19 | 94.61 | 60.0 | 95.0 | 189.634 | 97.0 | 0.4 |
| 1296000 | 94.67 | 94.61 | 75.0 | 95.0 | 188.084 | 95.0 | 0.4 |
| 1297000 | 94.89 | 94.62 | 87.0 | 95.0 | 191.392 | 98.0 | 0.4 |
| 1298000 | 94.89 | 94.62 | 84.0 | 95.0 | 192.447 | 99.0 | 0.4 |
| 1299000 | 94.78 | 94.65 | 78.0 | 95.0 | 191.344 | 98.0 | 0.4 |
| 1300000 | 94.72 | 94.65 | 76.0 | 95.0 | 191.265 | 98.0 | 0.4 |
