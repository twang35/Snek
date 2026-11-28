# b54g-ngu1mix-seed7

step **629,000** · 629 evals · trailing **94.4** · peak **94.99** @308,000 · sef **90.6** · best30 **99.9** @308,000

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
| seed | 7 |
| target_update_period | 1500 |
| torch_threads | 1 |

![b54g-ngu1mix-seed7](b54g-ngu1mix-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.47 | 1.47 | 0.0 | 7.0 | -0.805 | 0.0 | 0.4 |
| 2000 | 15.18 | 8.32 | 0.0 | 30.0 | 12.086 | 0.0 | 0.4 |
| 3000 | 48.81 | 21.82 | 16.0 | 88.0 | 43.537 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 618000 | 93.59 | 94.61 | 18.0 | 95.0 | 190.29 | 98.0 | 0.4 |
| 619000 | 94.18 | 94.59 | 22.0 | 95.0 | 190.883 | 98.0 | 0.4 |
| 620000 | 93.55 | 94.54 | 18.0 | 95.0 | 190.249 | 98.0 | 0.4 |
| 621000 | 94.31 | 94.54 | 26.0 | 95.0 | 192.059 | 99.0 | 0.4 |
| 622000 | 93.5 | 94.49 | 17.0 | 95.0 | 190.207 | 98.0 | 0.4 |
| 623000 | 94.17 | 94.47 | 15.0 | 95.0 | 190.873 | 98.0 | 0.4 |
| 624000 | 94.31 | 94.45 | 26.0 | 95.0 | 192.047 | 99.0 | 0.4 |
| 625000 | 94.09 | 94.42 | 16.0 | 95.0 | 189.833 | 97.0 | 0.4 |
| 626000 | 95.0 | 94.42 | 95.0 | 95.0 | 193.777 | 100.0 | 0.4 |
| 627000 | 94.26 | 94.42 | 21.0 | 95.0 | 192.002 | 99.0 | 0.4 |
| 628000 | 94.12 | 94.39 | 57.0 | 95.0 | 189.831 | 97.0 | 0.4 |
| 629000 | 94.35 | 94.4 | 30.0 | 95.0 | 192.09 | 99.0 | 0.4 |
