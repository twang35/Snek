# b53c-ngu-seed3

step **1,149,000** · 1149 evals · trailing **94.69** · peak **94.93** @574,000 · sef **80.1** · best30 **99.6** @585,000

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
| 1138000 | 94.98 | 94.71 | 94.0 | 95.0 | 191.604 | 98.0 | 0.4 |
| 1139000 | 94.21 | 94.71 | 52.0 | 95.0 | 189.803 | 97.0 | 0.4 |
| 1140000 | 93.87 | 94.68 | 15.0 | 95.0 | 187.426 | 95.0 | 0.4 |
| 1141000 | 94.9 | 94.69 | 89.0 | 95.0 | 191.525 | 98.0 | 0.4 |
| 1142000 | 95.0 | 94.72 | 95.0 | 95.0 | 193.72 | 100.0 | 0.4 |
| 1143000 | 93.44 | 94.71 | 16.0 | 95.0 | 188.028 | 96.0 | 0.4 |
| 1144000 | 94.81 | 94.7 | 88.0 | 95.0 | 190.383 | 97.0 | 0.4 |
| 1145000 | 94.54 | 94.69 | 82.0 | 95.0 | 186.981 | 94.0 | 0.4 |
| 1146000 | 94.69 | 94.7 | 64.0 | 95.0 | 192.358 | 99.0 | 0.4 |
| 1147000 | 94.97 | 94.7 | 92.0 | 95.0 | 192.681 | 99.0 | 0.4 |
| 1148000 | 95.0 | 94.7 | 95.0 | 95.0 | 193.714 | 100.0 | 0.4 |
| 1149000 | 94.89 | 94.69 | 87.0 | 95.0 | 191.535 | 98.0 | 0.4 |
