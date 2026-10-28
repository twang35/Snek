# b53d-ngu-seed4

step **1,259,000** · 1259 evals · trailing **94.63** · peak **94.94** @1,040,000 · sef **76.0** · best30 **99.7** @1,024,000

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
| seed | 4 |
| target_update_period | 1500 |
| torch_threads | 1 |

![b53d-ngu-seed4](b53d-ngu-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.54 | 0.54 | 0.0 | 3.0 | -4.486 | 0.0 | 0.4 |
| 2000 | 2.39 | 1.47 | 0.0 | 8.0 | 1.286 | 0.0 | 0.4 |
| 3000 | 6.99 | 3.31 | 0.0 | 25.0 | 5.34 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 1248000 | 94.98 | 94.73 | 93.0 | 95.0 | 192.644 | 99.0 | 0.4 |
| 1249000 | 94.95 | 94.73 | 90.0 | 95.0 | 192.625 | 99.0 | 0.4 |
| 1250000 | 94.3 | 94.7 | 25.0 | 95.0 | 191.965 | 99.0 | 0.4 |
| 1251000 | 95.0 | 94.71 | 95.0 | 95.0 | 193.714 | 100.0 | 0.4 |
| 1252000 | 94.96 | 94.73 | 92.0 | 95.0 | 191.613 | 98.0 | 0.4 |
| 1253000 | 94.97 | 94.73 | 92.0 | 95.0 | 192.67 | 99.0 | 0.4 |
| 1254000 | 93.49 | 94.7 | 22.0 | 95.0 | 189.124 | 97.0 | 0.4 |
| 1255000 | 95.0 | 94.7 | 95.0 | 95.0 | 193.715 | 100.0 | 0.4 |
| 1256000 | 94.35 | 94.68 | 32.0 | 95.0 | 191.011 | 98.0 | 0.4 |
| 1257000 | 94.14 | 94.65 | 24.0 | 95.0 | 190.772 | 98.0 | 0.4 |
| 1258000 | 94.17 | 94.63 | 24.0 | 95.0 | 189.846 | 97.0 | 0.4 |
| 1259000 | 93.49 | 94.63 | 21.0 | 95.0 | 189.108 | 97.0 | 0.4 |
