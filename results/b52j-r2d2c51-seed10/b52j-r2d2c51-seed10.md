# b52j-r2d2c51-seed10

step **1,800,000** · 1800 evals · trailing **94.8** · peak **94.86** @1,777,000 · sef **23.6** · best30 **98.5** @1,777,000

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
| max_steps | 1800000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.0 |
| n_step_update | 5 |
| priority_exponent | 0.9 |
| r2d2_burn_in | 40 |
| r2d2_head | c51 |
| r2d2_hidden | 512 |
| r2d2_is_beta | 0.6 |
| r2d2_prev_input | True |
| r2d2_priority_eta | 0.9 |
| r2d2_recurrent | lstm |
| r2d2_rescale | False |
| r2d2_rescale_eps | 0.001 |
| r2d2_seq_length | 120 |
| r2d2_stream_width | 512 |
| r2d2_stride | 40 |
| r2d2_windows | 30000 |
| replay_ratio | 0.003125 |
| seed | 10 |
| target_update_period | 2500 |
| torch_threads | 1 |

## Resumes

Resumed at 650,000, 1,170,000

![b52j-r2d2c51-seed10](b52j-r2d2c51-seed10.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.03 | 0.03 | 0.0 | 1.0 | -4.971 | 0.0 | 0.4 |
| 2000 | 0.02 | 0.03 | 0.0 | 1.0 | -4.982 | 0.0 | 0.4 |
| 3000 | 0.1 | 0.05 | 0.0 | 1.0 | -0.674 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1789000 | 94.97 | 94.83 | 92.0 | 95.0 | 192.614 | 99.0 | 0.4 |
| 1790000 | 94.9 | 94.83 | 88.0 | 95.0 | 191.487 | 98.0 | 0.4 |
| 1791000 | 94.2 | 94.81 | 36.0 | 95.0 | 189.805 | 97.0 | 0.4 |
| 1792000 | 94.82 | 94.8 | 90.0 | 95.0 | 188.419 | 95.0 | 0.4 |
| 1793000 | 94.96 | 94.83 | 91.0 | 95.0 | 192.62 | 99.0 | 0.4 |
| 1794000 | 95.0 | 94.83 | 95.0 | 95.0 | 193.662 | 100.0 | 0.4 |
| 1795000 | 94.66 | 94.84 | 61.0 | 95.0 | 192.273 | 99.0 | 0.4 |
| 1796000 | 94.81 | 94.83 | 76.0 | 95.0 | 192.407 | 99.0 | 0.4 |
| 1797000 | 95.0 | 94.84 | 95.0 | 95.0 | 193.634 | 100.0 | 0.4 |
| 1798000 | 94.8 | 94.83 | 84.0 | 95.0 | 190.415 | 97.0 | 0.4 |
| 1799000 | 94.61 | 94.82 | 70.0 | 95.0 | 189.156 | 96.0 | 0.4 |
| 1800000 | 94.09 | 94.8 | 4.0 | 95.0 | 191.697 | 99.0 | 0.4 |
