# b53a-ngu-seed1

step **1,279,000** · 1279 evals · trailing **94.66** · peak **94.96** @643,000 · sef **78.0** · best30 **99.7** @643,000

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
| 1268000 | 94.66 | 94.63 | 68.0 | 95.0 | 190.309 | 97.0 | 0.4 |
| 1269000 | 94.91 | 94.63 | 89.0 | 95.0 | 191.52 | 98.0 | 0.4 |
| 1270000 | 94.57 | 94.62 | 54.0 | 95.0 | 191.233 | 98.0 | 0.4 |
| 1271000 | 95.0 | 94.63 | 95.0 | 95.0 | 193.696 | 100.0 | 0.4 |
| 1272000 | 94.84 | 94.63 | 79.0 | 95.0 | 192.529 | 99.0 | 0.4 |
| 1273000 | 94.9 | 94.65 | 85.0 | 95.0 | 192.565 | 99.0 | 0.4 |
| 1274000 | 94.03 | 94.62 | 7.0 | 95.0 | 189.603 | 97.0 | 0.4 |
| 1275000 | 94.46 | 94.64 | 63.0 | 95.0 | 190.098 | 97.0 | 0.4 |
| 1276000 | 94.26 | 94.63 | 40.0 | 95.0 | 189.825 | 97.0 | 0.4 |
| 1277000 | 94.86 | 94.66 | 84.0 | 95.0 | 191.515 | 98.0 | 0.4 |
| 1278000 | 94.91 | 94.66 | 86.0 | 95.0 | 192.569 | 99.0 | 0.4 |
| 1279000 | 95.0 | 94.66 | 95.0 | 95.0 | 193.707 | 100.0 | 0.4 |
