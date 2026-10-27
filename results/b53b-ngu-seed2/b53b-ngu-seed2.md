# b53b-ngu-seed2

step **979,000** · 979 evals · trailing **94.89** · peak **94.95** @633,000 · sef **68.1** · best30 **99.6** @633,000

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
| seed | 2 |
| target_update_period | 1500 |
| torch_threads | 1 |

![b53b-ngu-seed2](b53b-ngu-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.41 | 1.41 | 0.0 | 5.0 | -1.796 | 0.0 | 0.4 |
| 2000 | 6.04 | 3.73 | 0.0 | 19.0 | 3.28 | 0.0 | 0.4 |
| 3000 | 18.06 | 8.5 | 1.0 | 40.0 | 15.298 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 968000 | 93.84 | 94.89 | 24.0 | 95.0 | 190.466 | 98.0 | 0.4 |
| 969000 | 95.0 | 94.89 | 95.0 | 95.0 | 193.719 | 100.0 | 0.4 |
| 970000 | 95.0 | 94.89 | 95.0 | 95.0 | 193.713 | 100.0 | 0.4 |
| 971000 | 95.0 | 94.89 | 95.0 | 95.0 | 193.715 | 100.0 | 0.4 |
| 972000 | 95.0 | 94.91 | 95.0 | 95.0 | 193.695 | 100.0 | 0.4 |
| 973000 | 95.0 | 94.91 | 95.0 | 95.0 | 193.709 | 100.0 | 0.4 |
| 974000 | 94.97 | 94.91 | 92.0 | 95.0 | 192.667 | 99.0 | 0.4 |
| 975000 | 95.0 | 94.91 | 95.0 | 95.0 | 193.717 | 100.0 | 0.4 |
| 976000 | 94.37 | 94.89 | 32.0 | 95.0 | 192.05 | 99.0 | 0.4 |
| 977000 | 95.0 | 94.89 | 95.0 | 95.0 | 193.704 | 100.0 | 0.4 |
| 978000 | 95.0 | 94.89 | 95.0 | 95.0 | 193.716 | 100.0 | 0.4 |
| 979000 | 95.0 | 94.89 | 95.0 | 95.0 | 193.711 | 100.0 | 0.4 |
