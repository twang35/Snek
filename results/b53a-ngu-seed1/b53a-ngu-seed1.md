# b53a-ngu-seed1

step **909,000** · 909 evals · trailing **94.75** · peak **94.96** @643,000 · sef **69.1** · best30 **99.7** @643,000

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
| 898000 | 94.72 | 94.82 | 74.0 | 95.0 | 191.423 | 98.0 | 0.4 |
| 899000 | 94.87 | 94.81 | 82.0 | 95.0 | 192.577 | 99.0 | 0.4 |
| 900000 | 94.92 | 94.81 | 87.0 | 95.0 | 192.564 | 99.0 | 0.4 |
| 901000 | 94.22 | 94.8 | 17.0 | 95.0 | 191.877 | 99.0 | 0.4 |
| 902000 | 94.93 | 94.8 | 88.0 | 95.0 | 192.58 | 99.0 | 0.4 |
| 903000 | 94.97 | 94.8 | 92.0 | 95.0 | 192.623 | 99.0 | 0.4 |
| 904000 | 94.21 | 94.77 | 16.0 | 95.0 | 191.856 | 99.0 | 0.4 |
| 905000 | 95.0 | 94.77 | 95.0 | 95.0 | 193.694 | 100.0 | 0.4 |
| 906000 | 94.95 | 94.77 | 90.0 | 95.0 | 192.61 | 99.0 | 0.4 |
| 907000 | 94.3 | 94.74 | 25.0 | 95.0 | 191.96 | 99.0 | 0.4 |
| 908000 | 95.0 | 94.74 | 95.0 | 95.0 | 193.664 | 100.0 | 0.4 |
| 909000 | 94.78 | 94.75 | 73.0 | 95.0 | 192.456 | 99.0 | 0.4 |
