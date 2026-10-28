# b53c-ngu-seed3

step **1,099,000** · 1099 evals · trailing **94.8** · peak **94.93** @574,000 · sef **79.2** · best30 **99.6** @585,000

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
| seed | 3 |
| target_update_period | 1500 |
| torch_threads | 1 |

![b53c-ngu-seed3](b53c-ngu-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.57 | 0.57 | 0.0 | 4.0 | 0.017 | 0.0 | 0.4 |
| 2000 | 6.19 | 3.38 | 0.0 | 14.0 | 2.147 | 0.0 | 0.4 |
| 3000 | 12.02 | 6.26 | 1.0 | 18.0 | 7.496 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1088000 | 95.0 | 94.91 | 95.0 | 95.0 | 193.711 | 100.0 | 0.4 |
| 1089000 | 94.95 | 94.91 | 92.0 | 95.0 | 191.596 | 98.0 | 0.4 |
| 1090000 | 95.0 | 94.91 | 95.0 | 95.0 | 193.706 | 100.0 | 0.4 |
| 1091000 | 94.23 | 94.88 | 20.0 | 95.0 | 190.867 | 98.0 | 0.4 |
| 1092000 | 94.94 | 94.89 | 90.0 | 95.0 | 191.571 | 98.0 | 0.4 |
| 1093000 | 94.46 | 94.87 | 44.0 | 95.0 | 190.042 | 97.0 | 0.4 |
| 1094000 | 95.0 | 94.89 | 95.0 | 95.0 | 193.699 | 100.0 | 0.4 |
| 1095000 | 94.64 | 94.88 | 64.0 | 95.0 | 190.246 | 97.0 | 0.4 |
| 1096000 | 94.82 | 94.88 | 85.0 | 95.0 | 190.386 | 97.0 | 0.4 |
| 1097000 | 94.9 | 94.87 | 88.0 | 95.0 | 191.558 | 98.0 | 0.4 |
| 1098000 | 93.5 | 94.82 | 16.0 | 95.0 | 190.115 | 98.0 | 0.4 |
| 1099000 | 94.23 | 94.8 | 23.0 | 95.0 | 188.762 | 96.0 | 0.4 |
