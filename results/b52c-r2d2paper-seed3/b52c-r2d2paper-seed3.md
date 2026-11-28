# b52c-r2d2paper-seed3

step **1,300,000** · 1300 evals · trailing **94.76** · peak **94.9** @995,000 · sef **60.2** · best30 **98.8** @1,079,000

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
| seed | 3 |
| target_update_period | 2500 |
| torch_threads | 1 |

![b52c-r2d2paper-seed3](b52c-r2d2paper-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.13 | 0.13 | 0.0 | 1.0 | -4.872 | 0.0 | 0.4 |
| 2000 | 0.29 | 0.21 | 0.0 | 2.0 | -0.263 | 0.0 | 0.4 |
| 3000 | 0.16 | 0.19 | 0.0 | 1.0 | -0.389 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1289000 | 95.0 | 94.8 | 95.0 | 95.0 | 193.692 | 100.0 | 0.4 |
| 1290000 | 94.56 | 94.79 | 80.0 | 95.0 | 188.262 | 95.0 | 0.4 |
| 1291000 | 94.99 | 94.79 | 94.0 | 95.0 | 192.634 | 99.0 | 0.4 |
| 1292000 | 94.53 | 94.79 | 52.0 | 95.0 | 191.153 | 98.0 | 0.4 |
| 1293000 | 95.0 | 94.8 | 95.0 | 95.0 | 193.694 | 100.0 | 0.4 |
| 1294000 | 94.88 | 94.8 | 83.0 | 95.0 | 192.576 | 99.0 | 0.4 |
| 1295000 | 94.65 | 94.79 | 60.0 | 95.0 | 192.361 | 99.0 | 0.4 |
| 1296000 | 94.97 | 94.79 | 92.0 | 95.0 | 192.626 | 99.0 | 0.4 |
| 1297000 | 94.73 | 94.78 | 84.0 | 95.0 | 190.374 | 97.0 | 0.4 |
| 1298000 | 93.81 | 94.76 | 50.0 | 95.0 | 189.475 | 97.0 | 0.4 |
| 1299000 | 94.75 | 94.75 | 82.0 | 95.0 | 190.36 | 97.0 | 0.4 |
| 1300000 | 94.84 | 94.76 | 86.0 | 95.0 | 191.54 | 98.0 | 0.4 |
