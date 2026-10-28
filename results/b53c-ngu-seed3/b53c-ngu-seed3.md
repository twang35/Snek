# b53c-ngu-seed3

step **1,249,000** · 1249 evals · trailing **94.73** · peak **94.93** @574,000 · sef **81.7** · best30 **99.6** @585,000

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
| 1238000 | 94.98 | 94.76 | 93.0 | 95.0 | 192.689 | 99.0 | 0.4 |
| 1239000 | 95.0 | 94.76 | 95.0 | 95.0 | 193.724 | 100.0 | 0.4 |
| 1240000 | 95.0 | 94.76 | 95.0 | 95.0 | 193.71 | 100.0 | 0.4 |
| 1241000 | 94.97 | 94.76 | 92.0 | 95.0 | 192.669 | 99.0 | 0.4 |
| 1242000 | 94.8 | 94.77 | 84.0 | 95.0 | 190.477 | 97.0 | 0.4 |
| 1243000 | 93.71 | 94.73 | 13.0 | 95.0 | 189.304 | 97.0 | 0.4 |
| 1244000 | 95.0 | 94.76 | 95.0 | 95.0 | 193.712 | 100.0 | 0.4 |
| 1245000 | 94.16 | 94.75 | 31.0 | 95.0 | 190.837 | 98.0 | 0.4 |
| 1246000 | 94.46 | 94.73 | 55.0 | 95.0 | 191.095 | 98.0 | 0.4 |
| 1247000 | 94.97 | 94.73 | 92.0 | 95.0 | 192.683 | 99.0 | 0.4 |
| 1248000 | 95.0 | 94.73 | 95.0 | 95.0 | 193.693 | 100.0 | 0.4 |
| 1249000 | 94.95 | 94.73 | 90.0 | 95.0 | 192.614 | 99.0 | 0.4 |
