# b53f-ngubeta0-seed6

step **939,000** · 939 evals · trailing **94.01** · peak **94.99** @323,000 · sef **85.6** · best30 **99.8** @323,000

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
| 928000 | 93.55 | 93.94 | 19.0 | 95.0 | 190.256 | 98.0 | 0.4 |
| 929000 | 95.0 | 93.97 | 95.0 | 95.0 | 193.784 | 100.0 | 0.4 |
| 930000 | 94.35 | 94.04 | 30.0 | 95.0 | 192.09 | 99.0 | 0.4 |
| 931000 | 95.0 | 94.07 | 95.0 | 95.0 | 193.778 | 100.0 | 0.4 |
| 932000 | 92.78 | 94.02 | 12.0 | 95.0 | 188.449 | 97.0 | 0.4 |
| 933000 | 94.24 | 94.03 | 19.0 | 95.0 | 191.983 | 99.0 | 0.4 |
| 934000 | 93.0 | 93.99 | 19.0 | 95.0 | 188.672 | 97.0 | 0.4 |
| 935000 | 94.28 | 94.03 | 23.0 | 95.0 | 192.025 | 99.0 | 0.4 |
| 936000 | 94.22 | 94.04 | 17.0 | 95.0 | 191.968 | 99.0 | 0.4 |
| 937000 | 94.27 | 94.06 | 22.0 | 95.0 | 192.02 | 99.0 | 0.4 |
| 938000 | 94.23 | 94.03 | 18.0 | 95.0 | 191.977 | 99.0 | 0.4 |
| 939000 | 93.51 | 94.01 | 11.0 | 95.0 | 190.221 | 98.0 | 0.4 |
