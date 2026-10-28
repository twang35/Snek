# b53c-ngu-seed3

step **1,179,000** · 1179 evals · trailing **94.68** · peak **94.93** @574,000 · sef **80.6** · best30 **99.6** @585,000

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
| 1168000 | 94.85 | 94.66 | 80.0 | 95.0 | 192.514 | 99.0 | 0.4 |
| 1169000 | 94.33 | 94.66 | 28.0 | 95.0 | 192.005 | 99.0 | 0.4 |
| 1170000 | 94.95 | 94.7 | 90.0 | 95.0 | 192.623 | 99.0 | 0.4 |
| 1171000 | 94.64 | 94.69 | 79.0 | 95.0 | 188.214 | 95.0 | 0.4 |
| 1172000 | 94.36 | 94.67 | 44.0 | 95.0 | 190.999 | 98.0 | 0.4 |
| 1173000 | 93.66 | 94.67 | 40.0 | 95.0 | 187.17 | 95.0 | 0.4 |
| 1174000 | 95.0 | 94.68 | 95.0 | 95.0 | 193.716 | 100.0 | 0.4 |
| 1175000 | 94.85 | 94.69 | 86.0 | 95.0 | 190.438 | 97.0 | 0.4 |
| 1176000 | 94.9 | 94.7 | 89.0 | 95.0 | 191.53 | 98.0 | 0.4 |
| 1177000 | 94.84 | 94.69 | 85.0 | 95.0 | 190.422 | 97.0 | 0.4 |
| 1178000 | 94.64 | 94.68 | 59.0 | 95.0 | 192.301 | 99.0 | 0.4 |
| 1179000 | 94.91 | 94.68 | 86.0 | 95.0 | 192.578 | 99.0 | 0.4 |
