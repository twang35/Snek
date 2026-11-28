# b54c-r2d2nguvals-seed3

step **700,000** · 700 evals · trailing **94.66** · peak **94.81** @657,000 · sef **63.4** · best30 **98.1** @697,000

## Config

| | |
|---|---|
| adam_epsilon | 0.0001 |
| algo | r2d2 |
| apex_alpha | 8.0 |
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
| max_steps | 700000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.0 |
| n_step_update | 5 |
| priority_exponent | 0.9 |
| r2d2_burn_in | 40 |
| r2d2_head | scalar |
| r2d2_hidden | 512 |
| r2d2_is_beta | 0.0 |
| r2d2_prev_input | True |
| r2d2_priority_eta | 1.0 |
| r2d2_recurrent | lstm |
| r2d2_rescale | True |
| r2d2_rescale_eps | 0.001 |
| r2d2_seq_length | 120 |
| r2d2_stream_width | 512 |
| r2d2_stride | 40 |
| r2d2_windows | 30000 |
| replay_ratio | 0.003125 |
| seed | 3 |
| target_update_period | 1500 |
| torch_threads | 1 |

![b54c-r2d2nguvals-seed3](b54c-r2d2nguvals-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.64 | 1.64 | 0.0 | 9.0 | 0.37 | 0.0 | 0.4 |
| 2000 | 8.77 | 5.21 | 1.0 | 14.0 | 3.835 | 0.0 | 0.4 |
| 3000 | 17.04 | 9.15 | 3.0 | 40.0 | 13.137 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 689000 | 94.48 | 94.65 | 50.0 | 95.0 | 191.062 | 98.0 | 0.4 |
| 690000 | 95.0 | 94.66 | 95.0 | 95.0 | 193.669 | 100.0 | 0.4 |
| 691000 | 94.81 | 94.66 | 86.0 | 95.0 | 190.368 | 97.0 | 0.4 |
| 692000 | 94.03 | 94.63 | 0.0 | 95.0 | 190.653 | 98.0 | 0.4 |
| 693000 | 94.72 | 94.66 | 82.0 | 95.0 | 190.324 | 97.0 | 0.4 |
| 694000 | 94.31 | 94.64 | 26.0 | 95.0 | 191.913 | 99.0 | 0.4 |
| 695000 | 94.54 | 94.64 | 57.0 | 95.0 | 190.204 | 97.0 | 0.4 |
| 696000 | 95.0 | 94.64 | 95.0 | 95.0 | 193.65 | 100.0 | 0.4 |
| 697000 | 94.95 | 94.65 | 90.0 | 95.0 | 192.583 | 99.0 | 0.4 |
| 698000 | 94.14 | 94.62 | 27.0 | 95.0 | 189.686 | 97.0 | 0.4 |
| 699000 | 94.23 | 94.61 | 37.0 | 95.0 | 188.721 | 96.0 | 0.4 |
| 700000 | 94.99 | 94.66 | 94.0 | 95.0 | 192.607 | 99.0 | 0.4 |
