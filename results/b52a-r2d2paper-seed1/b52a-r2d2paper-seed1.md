# b52a-r2d2paper-seed1

step **1,300,000** · 1300 evals · trailing **94.63** · peak **94.89** @998,000 · sef **65.8** · best30 **99.0** @1,017,000

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
| seed | 1 |
| target_update_period | 2500 |
| torch_threads | 1 |

![b52a-r2d2paper-seed1](b52a-r2d2paper-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.05 | 0.05 | 0.0 | 1.0 | -4.953 | 0.0 | 0.4 |
| 2000 | 0.78 | 0.42 | 0.0 | 3.0 | -0.174 | 0.0 | 0.4 |
| 3000 | 6.88 | 2.57 | 0.0 | 13.0 | 2.835 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1289000 | 94.32 | 94.68 | 62.0 | 95.0 | 187.938 | 95.0 | 0.4 |
| 1290000 | 94.67 | 94.68 | 62.0 | 95.0 | 192.357 | 99.0 | 0.4 |
| 1291000 | 94.69 | 94.68 | 64.0 | 95.0 | 192.391 | 99.0 | 0.4 |
| 1292000 | 94.67 | 94.68 | 62.0 | 95.0 | 192.367 | 99.0 | 0.4 |
| 1293000 | 94.95 | 94.69 | 90.0 | 95.0 | 192.646 | 99.0 | 0.4 |
| 1294000 | 94.96 | 94.69 | 93.0 | 95.0 | 191.609 | 98.0 | 0.4 |
| 1295000 | 95.0 | 94.7 | 95.0 | 95.0 | 193.687 | 100.0 | 0.4 |
| 1296000 | 94.4 | 94.7 | 35.0 | 95.0 | 192.087 | 99.0 | 0.4 |
| 1297000 | 95.0 | 94.7 | 95.0 | 95.0 | 193.668 | 100.0 | 0.4 |
| 1298000 | 94.42 | 94.68 | 50.0 | 95.0 | 191.108 | 98.0 | 0.4 |
| 1299000 | 94.32 | 94.66 | 59.0 | 95.0 | 191.009 | 98.0 | 0.4 |
| 1300000 | 94.19 | 94.63 | 60.0 | 95.0 | 188.845 | 96.0 | 0.4 |
