# b53b-ngu-seed2

step **1,059,000** · 1059 evals · trailing **94.58** · peak **94.96** @1,028,000 · sef **70.5** · best30 **99.8** @983,000

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
| seed | 2 |
| target_update_period | 1500 |
| torch_threads | 1 |

![b53b-ngu-seed2](b53b-ngu-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.41 | 1.41 | 0.0 | 5.0 | -1.796 | 0.0 | 0.4 |
| 2000 | 6.04 | 3.73 | 0.0 | 19.0 | 3.28 | 0.0 | 0.4 |
| 3000 | 18.06 | 8.5 | 1.0 | 40.0 | 15.298 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1048000 | 94.44 | 94.85 | 39.0 | 95.0 | 192.12 | 99.0 | 0.4 |
| 1049000 | 94.91 | 94.85 | 90.0 | 95.0 | 191.557 | 98.0 | 0.4 |
| 1050000 | 94.34 | 94.83 | 29.0 | 95.0 | 192.016 | 99.0 | 0.4 |
| 1051000 | 92.5 | 94.74 | 17.0 | 95.0 | 186.044 | 95.0 | 0.4 |
| 1052000 | 94.91 | 94.74 | 86.0 | 95.0 | 192.579 | 99.0 | 0.4 |
| 1053000 | 95.0 | 94.74 | 95.0 | 95.0 | 193.718 | 100.0 | 0.4 |
| 1054000 | 92.64 | 94.66 | 4.0 | 95.0 | 188.331 | 97.0 | 0.4 |
| 1055000 | 93.89 | 94.63 | 38.0 | 95.0 | 189.524 | 97.0 | 0.4 |
| 1056000 | 95.0 | 94.63 | 95.0 | 95.0 | 193.728 | 100.0 | 0.4 |
| 1057000 | 95.0 | 94.63 | 95.0 | 95.0 | 193.721 | 100.0 | 0.4 |
| 1058000 | 93.36 | 94.58 | 8.0 | 95.0 | 190.009 | 98.0 | 0.4 |
| 1059000 | 95.0 | 94.58 | 95.0 | 95.0 | 193.712 | 100.0 | 0.4 |
