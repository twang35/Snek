# b52d-r2d2paper-seed4

step **1,300,000** · 1300 evals · trailing **94.68** · peak **94.82** @1,179,000 · sef **43.9** · best30 **97.8** @1,180,000

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
| seed | 4 |
| target_update_period | 2500 |
| torch_threads | 1 |

![b52d-r2d2paper-seed4](b52d-r2d2paper-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.08 | 0.08 | 0.0 | 1.0 | -4.923 | 0.0 | 0.4 |
| 2000 | 2.66 | 1.37 | 0.0 | 5.0 | -1.524 | 0.0 | 0.4 |
| 3000 | 3.22 | 1.99 | 0.0 | 6.0 | -0.472 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1289000 | 94.81 | 94.68 | 84.0 | 95.0 | 190.35 | 97.0 | 0.4 |
| 1290000 | 94.58 | 94.67 | 76.0 | 95.0 | 190.212 | 97.0 | 0.4 |
| 1291000 | 94.78 | 94.67 | 78.0 | 95.0 | 191.363 | 98.0 | 0.4 |
| 1292000 | 94.73 | 94.67 | 88.0 | 95.0 | 186.263 | 93.0 | 0.4 |
| 1293000 | 94.54 | 94.67 | 70.0 | 95.0 | 187.188 | 94.0 | 0.4 |
| 1294000 | 94.81 | 94.67 | 88.0 | 95.0 | 190.422 | 97.0 | 0.4 |
| 1295000 | 94.42 | 94.67 | 76.0 | 95.0 | 183.913 | 91.0 | 0.4 |
| 1296000 | 94.41 | 94.65 | 72.0 | 95.0 | 185.015 | 92.0 | 0.4 |
| 1297000 | 94.93 | 94.66 | 92.0 | 95.0 | 190.487 | 97.0 | 0.4 |
| 1298000 | 94.79 | 94.66 | 85.0 | 95.0 | 189.41 | 96.0 | 0.4 |
| 1299000 | 94.89 | 94.67 | 86.0 | 95.0 | 191.522 | 98.0 | 0.4 |
| 1300000 | 94.61 | 94.68 | 84.0 | 95.0 | 186.183 | 93.0 | 0.4 |
