# b54h-ngu1mix-seed8

step **700,000** · 700 evals · trailing **94.44** · peak **94.99** @332,000 · sef **76.1** · best30 **99.9** @424,000

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
| seed | 8 |
| target_update_period | 1500 |
| torch_threads | 1 |

![b54h-ngu1mix-seed8](b54h-ngu1mix-seed8.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.65 | 1.65 | 0.0 | 9.0 | 0.625 | 0.0 | 0.4 |
| 2000 | 29.66 | 15.65 | 10.0 | 60.0 | 24.462 | 0.0 | 0.4 |
| 3000 | 55.64 | 28.98 | 19.0 | 90.0 | 50.395 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 689000 | 94.25 | 94.83 | 20.0 | 95.0 | 191.986 | 99.0 | 0.4 |
| 690000 | 93.41 | 94.78 | 24.0 | 95.0 | 189.068 | 97.0 | 0.4 |
| 691000 | 95.0 | 94.79 | 95.0 | 95.0 | 193.778 | 100.0 | 0.4 |
| 692000 | 92.93 | 94.72 | 24.0 | 95.0 | 188.585 | 97.0 | 0.4 |
| 693000 | 95.0 | 94.75 | 95.0 | 95.0 | 193.769 | 100.0 | 0.4 |
| 694000 | 93.55 | 94.73 | 20.0 | 95.0 | 190.254 | 98.0 | 0.4 |
| 695000 | 93.4 | 94.67 | 13.0 | 95.0 | 190.097 | 98.0 | 0.4 |
| 696000 | 94.27 | 94.65 | 22.0 | 95.0 | 191.997 | 99.0 | 0.4 |
| 697000 | 93.55 | 94.6 | 22.0 | 95.0 | 190.243 | 98.0 | 0.4 |
| 698000 | 93.41 | 94.55 | 14.0 | 95.0 | 190.113 | 98.0 | 0.4 |
| 699000 | 93.44 | 94.5 | 11.0 | 95.0 | 190.147 | 98.0 | 0.4 |
| 700000 | 93.25 | 94.44 | 15.0 | 95.0 | 187.956 | 96.0 | 0.4 |
