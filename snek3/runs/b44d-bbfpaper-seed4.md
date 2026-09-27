# b44d-bbfpaper-seed4

step **100,000** · 100 evals · trailing **22.58** · peak **30.97** @9,000 · sef **0.0** · best30 **0.0** @100,000

## Config

| | |
|---|---|
| adam_epsilon | 0.00015 |
| algo | bbf |
| batch_size | 32 |
| bbf_double | True |
| bbf_dueling | True |
| bbf_projection | 512 |
| bbf_spr_steps | 5 |
| bbf_spr_weight | 5.0 |
| bbf_transition_width | 256 |
| bbf_weight_decay | 0.1 |
| collect_envs | 1 |
| discount | 0.997 |
| dist_atoms | 51 |
| dist_v_max | 110.0 |
| dist_v_min | -10.0 |
| epsilon_anneal_steps | 2001 |
| eval_interval | 1000 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (1280, 2048) |
| gradient_clipping | 10.0 |
| graph_eval_episodes | 100 |
| init_from | None |
| initial_collect_steps | 2000 |
| initial_epsilon | 1.0 |
| learning_rate | 0.0001 |
| max_steps | 100000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.0 |
| n_step_update | 3 |
| priority_exponent | 0.5 |
| replay_buffer_max_length | 1000000 |
| replay_ratio | 8.0 |
| reset_alpha | 0.5 |
| reset_anneal_gamma | 0.97,0.997 |
| reset_anneal_n_step | 10,3 |
| reset_anneal_steps | 10000 |
| reset_interval | 40000 |
| reset_stop_after | 0 |
| seed | 4 |
| target_update_tau | 0.005 |
| torch_threads | 1 |

![b44d-bbfpaper-seed4](b44d-bbfpaper-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 9.58 | 9.58 | 0.0 | 23.0 | 4.728 | 0.0 | 0.50075 |
| 2000 | 7.2 | 8.39 | 2.0 | 18.0 | 3.719 | 0.0 | 0.001 |
| 3000 | 22.67 | 13.15 | 1.0 | 55.0 | 17.67 | 0.0 | 0.0 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 89000 | 43.81 | 19.14 | 3.0 | 74.0 | 41.148 | 0.0 | 0.0 |
| 90000 | 0.0 | 19.14 | 0.0 | 0.0 | -5.006 | 0.0 | 0.0 |
| 91000 | 44.52 | 19.76 | 13.0 | 76.0 | 39.611 | 0.0 | 0.0 |
| 92000 | 29.19 | 20.37 | 8.0 | 84.0 | 26.692 | 0.0 | 0.0 |
| 93000 | 22.62 | 20.72 | 0.0 | 53.0 | 21.189 | 0.0 | 0.0 |
| 94000 | 45.89 | 21.54 | 0.0 | 86.0 | 42.746 | 0.0 | 0.0 |
| 95000 | 0.0 | 21.54 | 0.0 | 0.0 | -5.001 | 0.0 | 0.0 |
| 96000 | 33.92 | 21.61 | 12.0 | 61.0 | 28.911 | 0.0 | 0.0 |
| 97000 | 11.37 | 21.63 | 2.0 | 21.0 | 8.261 | 0.0 | 0.0 |
| 98000 | 18.07 | 21.78 | 3.0 | 51.0 | 16.362 | 0.0 | 0.0 |
| 99000 | 43.64 | 22.58 | 3.0 | 78.0 | 40.128 | 0.0 | 0.0 |
| 100000 | 0.01 | 22.58 | 0.0 | 1.0 | -4.993 | 0.0 | 0.0 |
