# b54f-ngu1mix-seed6

step **689,000** · 689 evals · trailing **94.74** · peak **94.97** @342,000 · sef **84.8** · best30 **99.8** @342,000

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
| max_steps | 700000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.0 |
| n_step_update | 5 |
| ngu_assign | episode |
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
| ngu_mixtures | 1 |
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

![b54f-ngu1mix-seed6](b54f-ngu1mix-seed6.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.69 | 1.69 | 0.0 | 14.0 | 0.082 | 0.0 | 0.4 |
| 2000 | 34.39 | 18.04 | 1.0 | 64.0 | 29.959 | 0.0 | 0.4 |
| 3000 | 43.5 | 26.53 | 14.0 | 95.0 | 39.392 | 1.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 678000 | 95.0 | 94.7 | 95.0 | 95.0 | 193.773 | 100.0 | 0.4 |
| 679000 | 95.0 | 94.7 | 95.0 | 95.0 | 193.77 | 100.0 | 0.4 |
| 680000 | 94.17 | 94.72 | 12.0 | 95.0 | 191.91 | 99.0 | 0.4 |
| 681000 | 94.94 | 94.72 | 89.0 | 95.0 | 192.718 | 99.0 | 0.4 |
| 682000 | 95.0 | 94.72 | 95.0 | 95.0 | 193.781 | 100.0 | 0.4 |
| 683000 | 94.27 | 94.69 | 22.0 | 95.0 | 192.007 | 99.0 | 0.4 |
| 684000 | 94.32 | 94.67 | 27.0 | 95.0 | 192.057 | 99.0 | 0.4 |
| 685000 | 95.0 | 94.67 | 95.0 | 95.0 | 193.774 | 100.0 | 0.4 |
| 686000 | 95.0 | 94.67 | 95.0 | 95.0 | 193.779 | 100.0 | 0.4 |
| 687000 | 95.0 | 94.72 | 95.0 | 95.0 | 193.77 | 100.0 | 0.4 |
| 688000 | 94.24 | 94.7 | 19.0 | 95.0 | 191.977 | 99.0 | 0.4 |
| 689000 | 95.0 | 94.74 | 95.0 | 95.0 | 193.783 | 100.0 | 0.4 |
