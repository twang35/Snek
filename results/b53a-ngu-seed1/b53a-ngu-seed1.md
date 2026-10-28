# b53a-ngu-seed1

step **1,099,000** · 1099 evals · trailing **94.6** · peak **94.96** @643,000 · sef **74.4** · best30 **99.7** @643,000

## Config

| | |
|---|---|
| adam_epsilon | 0.0001 |
| algo | ngu |
| apex_alpha | 8.0 |
| batch_size | 64 |
| collect_envs | 32 |
| discount | 0.997 |
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
| ngu_assign | fixed |
| ngu_beta_max | 0.3 |
| ngu_cluster_distance | 0.008 |
| ngu_embed_dim | 32 |
| ngu_embed_hidden | 128 |
| ngu_embed_l2 | 1e-05 |
| ngu_embed_lr | 0.0005 |
| ngu_embed_steps | 5 |
| ngu_epsilon_permute | True |
| ngu_gamma_min | 0.99 |
| ngu_input | board |
| ngu_kernel_epsilon | 0.0001 |
| ngu_knn_k | 10 |
| ngu_lifelong_max | 5.0 |
| ngu_max_similarity | 8.0 |
| ngu_memory | 30000 |
| ngu_min_windows | 6250 |
| ngu_mixtures | 32 |
| ngu_pseudo_count | 0.001 |
| ngu_retrace_lambda | 0.95 |
| ngu_rnd_hidden | 128 |
| ngu_rnd_out | 128 |
| ngu_target_epsilon | 0.01 |
| priority_exponent | 0.9 |
| r2d2_burn_in | 40 |
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
| seed | 1 |
| target_update_period | 1500 |
| torch_threads | 1 |

![b53a-ngu-seed1](b53a-ngu-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.26 | 1.26 | 0.0 | 5.0 | -2.569 | 0.0 | 0.4 |
| 2000 | 8.43 | 4.84 | 0.0 | 21.0 | 7.201 | 0.0 | 0.4 |
| 3000 | 8.52 | 6.07 | 0.0 | 26.0 | 5.525 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1088000 | 94.99 | 94.53 | 94.0 | 95.0 | 192.634 | 99.0 | 0.4 |
| 1089000 | 95.0 | 94.53 | 95.0 | 95.0 | 193.695 | 100.0 | 0.4 |
| 1090000 | 94.98 | 94.54 | 93.0 | 95.0 | 192.629 | 99.0 | 0.4 |
| 1091000 | 94.28 | 94.52 | 23.0 | 95.0 | 191.923 | 99.0 | 0.4 |
| 1092000 | 94.92 | 94.52 | 89.0 | 95.0 | 191.506 | 98.0 | 0.4 |
| 1093000 | 95.0 | 94.52 | 95.0 | 95.0 | 193.675 | 100.0 | 0.4 |
| 1094000 | 95.0 | 94.53 | 95.0 | 95.0 | 193.701 | 100.0 | 0.4 |
| 1095000 | 94.95 | 94.53 | 90.0 | 95.0 | 192.633 | 99.0 | 0.4 |
| 1096000 | 94.42 | 94.56 | 52.0 | 95.0 | 190.042 | 97.0 | 0.4 |
| 1097000 | 94.12 | 94.56 | 11.0 | 95.0 | 189.743 | 97.0 | 0.4 |
| 1098000 | 94.96 | 94.57 | 91.0 | 95.0 | 192.602 | 99.0 | 0.4 |
| 1099000 | 95.0 | 94.6 | 95.0 | 95.0 | 193.714 | 100.0 | 0.4 |
