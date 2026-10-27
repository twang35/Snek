# b53a-ngu-seed1

step **899,000** · 899 evals · trailing **94.81** · peak **94.96** @643,000 · sef **68.7** · best30 **99.7** @643,000

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
| 888000 | 94.47 | 94.9 | 42.0 | 95.0 | 192.133 | 99.0 | 0.4 |
| 889000 | 95.0 | 94.9 | 95.0 | 95.0 | 193.712 | 100.0 | 0.4 |
| 890000 | 94.22 | 94.87 | 17.0 | 95.0 | 191.841 | 99.0 | 0.4 |
| 891000 | 94.59 | 94.86 | 54.0 | 95.0 | 192.252 | 99.0 | 0.4 |
| 892000 | 94.95 | 94.86 | 90.0 | 95.0 | 192.605 | 99.0 | 0.4 |
| 893000 | 95.0 | 94.86 | 95.0 | 95.0 | 193.697 | 100.0 | 0.4 |
| 894000 | 94.98 | 94.86 | 93.0 | 95.0 | 192.665 | 99.0 | 0.4 |
| 895000 | 94.15 | 94.83 | 10.0 | 95.0 | 191.805 | 99.0 | 0.4 |
| 896000 | 95.0 | 94.83 | 95.0 | 95.0 | 193.7 | 100.0 | 0.4 |
| 897000 | 94.49 | 94.81 | 44.0 | 95.0 | 192.147 | 99.0 | 0.4 |
| 898000 | 94.72 | 94.82 | 74.0 | 95.0 | 191.423 | 98.0 | 0.4 |
| 899000 | 94.87 | 94.81 | 82.0 | 95.0 | 192.577 | 99.0 | 0.4 |
