# b53f-ngubeta0-seed6

step **949,000** · 949 evals · trailing **94.19** · peak **94.99** @323,000 · sef **85.8** · best30 **99.8** @323,000

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
| 938000 | 94.23 | 94.03 | 18.0 | 95.0 | 191.977 | 99.0 | 0.4 |
| 939000 | 93.51 | 94.01 | 11.0 | 95.0 | 190.221 | 98.0 | 0.4 |
| 940000 | 95.0 | 94.06 | 95.0 | 95.0 | 193.795 | 100.0 | 0.4 |
| 941000 | 95.0 | 94.06 | 95.0 | 95.0 | 193.783 | 100.0 | 0.4 |
| 942000 | 94.21 | 94.06 | 16.0 | 95.0 | 191.951 | 99.0 | 0.4 |
| 943000 | 93.44 | 94.03 | 16.0 | 95.0 | 190.154 | 98.0 | 0.4 |
| 944000 | 95.0 | 94.09 | 95.0 | 95.0 | 193.784 | 100.0 | 0.4 |
| 945000 | 94.89 | 94.09 | 84.0 | 95.0 | 192.683 | 99.0 | 0.4 |
| 946000 | 94.25 | 94.09 | 20.0 | 95.0 | 191.995 | 99.0 | 0.4 |
| 947000 | 93.41 | 94.13 | 10.0 | 95.0 | 190.117 | 98.0 | 0.4 |
| 948000 | 93.51 | 94.14 | 18.0 | 95.0 | 190.207 | 98.0 | 0.4 |
| 949000 | 95.0 | 94.19 | 95.0 | 95.0 | 193.78 | 100.0 | 0.4 |
