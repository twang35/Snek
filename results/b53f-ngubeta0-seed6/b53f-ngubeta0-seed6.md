# b53f-ngubeta0-seed6

step **1,300,000** · 1300 evals · trailing **94.55** · peak **94.99** @323,000 · sef **89.6** · best30 **99.8** @323,000

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
| ngu_beta_max | 0.0 |
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
| seed | 6 |
| target_update_period | 1500 |
| torch_threads | 1 |

![b53f-ngubeta0-seed6](b53f-ngubeta0-seed6.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.14 | 1.14 | 0.0 | 11.0 | 0.209 | 0.0 | 0.4 |
| 2000 | 35.17 | 18.16 | 10.0 | 67.0 | 29.965 | 0.0 | 0.4 |
| 3000 | 47.63 | 27.98 | 1.0 | 93.0 | 42.642 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1289000 | 94.27 | 94.66 | 22.0 | 95.0 | 192.031 | 99.0 | 0.4 |
| 1290000 | 93.52 | 94.61 | 20.0 | 95.0 | 190.227 | 98.0 | 0.4 |
| 1291000 | 93.44 | 94.55 | 15.0 | 95.0 | 190.156 | 98.0 | 0.4 |
| 1292000 | 94.21 | 94.53 | 16.0 | 95.0 | 191.961 | 99.0 | 0.4 |
| 1293000 | 94.4 | 94.56 | 37.0 | 95.0 | 191.113 | 98.0 | 0.4 |
| 1294000 | 94.28 | 94.54 | 23.0 | 95.0 | 192.032 | 99.0 | 0.4 |
| 1295000 | 93.54 | 94.49 | 15.0 | 95.0 | 190.26 | 98.0 | 0.4 |
| 1296000 | 94.24 | 94.46 | 19.0 | 95.0 | 191.994 | 99.0 | 0.4 |
| 1297000 | 95.0 | 94.46 | 95.0 | 95.0 | 193.79 | 100.0 | 0.4 |
| 1298000 | 94.13 | 94.53 | 13.0 | 95.0 | 190.84 | 98.0 | 0.4 |
| 1299000 | 95.0 | 94.53 | 95.0 | 95.0 | 193.787 | 100.0 | 0.4 |
| 1300000 | 95.0 | 94.55 | 95.0 | 95.0 | 193.789 | 100.0 | 0.4 |
