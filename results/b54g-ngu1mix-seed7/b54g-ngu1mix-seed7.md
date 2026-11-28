# b54g-ngu1mix-seed7

step **639,000** · 639 evals · trailing **94.43** · peak **94.99** @308,000 · sef **90.8** · best30 **99.9** @308,000

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
| 628000 | 94.12 | 94.39 | 57.0 | 95.0 | 189.831 | 97.0 | 0.4 |
| 629000 | 94.35 | 94.4 | 30.0 | 95.0 | 192.09 | 99.0 | 0.4 |
| 630000 | 94.27 | 94.4 | 22.0 | 95.0 | 192.017 | 99.0 | 0.4 |
| 631000 | 94.2 | 94.4 | 15.0 | 95.0 | 191.943 | 99.0 | 0.4 |
| 632000 | 94.17 | 94.37 | 12.0 | 95.0 | 191.92 | 99.0 | 0.4 |
| 633000 | 94.27 | 94.35 | 34.0 | 95.0 | 191.017 | 98.0 | 0.4 |
| 634000 | 94.86 | 94.38 | 84.0 | 95.0 | 191.646 | 98.0 | 0.4 |
| 635000 | 95.0 | 94.4 | 95.0 | 95.0 | 193.776 | 100.0 | 0.4 |
| 636000 | 95.0 | 94.43 | 95.0 | 95.0 | 193.777 | 100.0 | 0.4 |
| 637000 | 94.24 | 94.4 | 19.0 | 95.0 | 191.984 | 99.0 | 0.4 |
| 638000 | 95.0 | 94.43 | 95.0 | 95.0 | 193.786 | 100.0 | 0.4 |
| 639000 | 95.0 | 94.43 | 95.0 | 95.0 | 193.783 | 100.0 | 0.4 |
