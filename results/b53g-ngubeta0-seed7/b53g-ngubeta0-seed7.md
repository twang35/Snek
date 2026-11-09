# b53g-ngubeta0-seed7

step **999,000** · 999 evals · trailing **94.46** · peak **94.97** @274,000 · sef **88.9** · best30 **99.8** @348,000

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
| 988000 | 93.44 | 94.43 | 16.0 | 95.0 | 190.16 | 98.0 | 0.4 |
| 989000 | 94.34 | 94.41 | 29.0 | 95.0 | 192.089 | 99.0 | 0.4 |
| 990000 | 93.5 | 94.38 | 16.0 | 95.0 | 190.221 | 98.0 | 0.4 |
| 991000 | 95.0 | 94.41 | 95.0 | 95.0 | 193.787 | 100.0 | 0.4 |
| 992000 | 94.18 | 94.43 | 13.0 | 95.0 | 191.925 | 99.0 | 0.4 |
| 993000 | 95.0 | 94.43 | 95.0 | 95.0 | 193.784 | 100.0 | 0.4 |
| 994000 | 95.0 | 94.45 | 95.0 | 95.0 | 193.79 | 100.0 | 0.4 |
| 995000 | 93.6 | 94.4 | 18.0 | 95.0 | 190.312 | 98.0 | 0.4 |
| 996000 | 95.0 | 94.51 | 95.0 | 95.0 | 193.794 | 100.0 | 0.4 |
| 997000 | 95.0 | 94.53 | 95.0 | 95.0 | 193.795 | 100.0 | 0.4 |
| 998000 | 92.01 | 94.46 | 8.0 | 95.0 | 186.649 | 96.0 | 0.4 |
| 999000 | 93.48 | 94.46 | 24.0 | 95.0 | 189.193 | 97.0 | 0.4 |
