# b53a-ngu-seed1

step **1,239,000** · 1239 evals · trailing **94.78** · peak **94.96** @643,000 · sef **77.3** · best30 **99.7** @643,000

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
| 1228000 | 94.72 | 94.77 | 79.0 | 95.0 | 191.34 | 98.0 | 0.4 |
| 1229000 | 94.94 | 94.77 | 92.0 | 95.0 | 191.567 | 98.0 | 0.4 |
| 1230000 | 94.66 | 94.77 | 61.0 | 95.0 | 192.358 | 99.0 | 0.4 |
| 1231000 | 94.06 | 94.76 | 9.0 | 95.0 | 189.694 | 97.0 | 0.4 |
| 1232000 | 94.91 | 94.78 | 87.0 | 95.0 | 191.535 | 98.0 | 0.4 |
| 1233000 | 94.98 | 94.79 | 93.0 | 95.0 | 192.628 | 99.0 | 0.4 |
| 1234000 | 94.01 | 94.8 | 12.0 | 95.0 | 189.573 | 97.0 | 0.4 |
| 1235000 | 94.85 | 94.8 | 80.0 | 95.0 | 192.513 | 99.0 | 0.4 |
| 1236000 | 94.73 | 94.79 | 68.0 | 95.0 | 192.4 | 99.0 | 0.4 |
| 1237000 | 94.81 | 94.78 | 76.0 | 95.0 | 192.473 | 99.0 | 0.4 |
| 1238000 | 95.0 | 94.78 | 95.0 | 95.0 | 193.689 | 100.0 | 0.4 |
| 1239000 | 94.8 | 94.78 | 85.0 | 95.0 | 189.376 | 96.0 | 0.4 |
