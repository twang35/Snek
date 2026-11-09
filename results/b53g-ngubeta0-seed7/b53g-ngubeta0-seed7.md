# b53g-ngubeta0-seed7

step **1,199,000** · 1199 evals · trailing **94.59** · peak **94.97** @274,000 · sef **90.7** · best30 **99.8** @1,111,000

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
| seed | 7 |
| target_update_period | 1500 |
| torch_threads | 1 |

![b53g-ngubeta0-seed7](b53g-ngubeta0-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 2.88 | 2.88 | 0.0 | 9.0 | 0.146 | 0.0 | 0.4 |
| 2000 | 37.8 | 20.34 | 6.0 | 63.0 | 32.628 | 0.0 | 0.4 |
| 3000 | 67.99 | 36.22 | 23.0 | 95.0 | 74.006 | 11.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1188000 | 94.17 | 94.65 | 12.0 | 95.0 | 191.93 | 99.0 | 0.4 |
| 1189000 | 95.0 | 94.65 | 95.0 | 95.0 | 193.787 | 100.0 | 0.4 |
| 1190000 | 94.33 | 94.62 | 28.0 | 95.0 | 192.086 | 99.0 | 0.4 |
| 1191000 | 95.0 | 94.62 | 95.0 | 95.0 | 193.786 | 100.0 | 0.4 |
| 1192000 | 95.0 | 94.62 | 95.0 | 95.0 | 193.788 | 100.0 | 0.4 |
| 1193000 | 92.88 | 94.58 | 21.0 | 95.0 | 188.562 | 97.0 | 0.4 |
| 1194000 | 95.0 | 94.6 | 95.0 | 95.0 | 193.788 | 100.0 | 0.4 |
| 1195000 | 95.0 | 94.67 | 95.0 | 95.0 | 193.791 | 100.0 | 0.4 |
| 1196000 | 95.0 | 94.67 | 95.0 | 95.0 | 193.797 | 100.0 | 0.4 |
| 1197000 | 94.41 | 94.65 | 36.0 | 95.0 | 192.166 | 99.0 | 0.4 |
| 1198000 | 94.16 | 94.62 | 11.0 | 95.0 | 191.917 | 99.0 | 0.4 |
| 1199000 | 94.16 | 94.59 | 11.0 | 95.0 | 191.91 | 99.0 | 0.4 |
