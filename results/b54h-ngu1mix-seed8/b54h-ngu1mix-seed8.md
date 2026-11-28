# b54h-ngu1mix-seed8

step **629,000** · 629 evals · trailing **94.8** · peak **94.99** @332,000 · sef **73.4** · best30 **99.9** @424,000

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
| 618000 | 95.0 | 94.86 | 95.0 | 95.0 | 193.769 | 100.0 | 0.4 |
| 619000 | 95.0 | 94.86 | 95.0 | 95.0 | 193.763 | 100.0 | 0.4 |
| 620000 | 94.97 | 94.86 | 92.0 | 95.0 | 192.705 | 99.0 | 0.4 |
| 621000 | 95.0 | 94.86 | 95.0 | 95.0 | 193.767 | 100.0 | 0.4 |
| 622000 | 94.23 | 94.83 | 18.0 | 95.0 | 191.953 | 99.0 | 0.4 |
| 623000 | 95.0 | 94.83 | 95.0 | 95.0 | 193.769 | 100.0 | 0.4 |
| 624000 | 95.0 | 94.83 | 95.0 | 95.0 | 193.766 | 100.0 | 0.4 |
| 625000 | 93.69 | 94.79 | 27.0 | 95.0 | 190.384 | 98.0 | 0.4 |
| 626000 | 94.22 | 94.79 | 19.0 | 95.0 | 190.905 | 98.0 | 0.4 |
| 627000 | 95.0 | 94.82 | 95.0 | 95.0 | 193.763 | 100.0 | 0.4 |
| 628000 | 94.33 | 94.8 | 29.0 | 95.0 | 191.004 | 98.0 | 0.4 |
| 629000 | 94.97 | 94.8 | 92.0 | 95.0 | 192.698 | 99.0 | 0.4 |
