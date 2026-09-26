# b47d-btrpaper-seed4

step **1,000,000** · 1000 evals · trailing **94.73** · peak **94.73** @1,000,000 · sef **45.5** · best30 **96.7** @1,000,000

## Config

| | |
|---|---|
| adam_epsilon | 1.953125e-05 |
| algo | btr |
| batch_size | 256 |
| beta_anneal_steps | 1 |
| btr_blocks | 3 |
| btr_layer_norm | False |
| btr_residual | True |
| btr_spectral_norm | True |
| collect_envs | 64 |
| discount | 0.997 |
| dist_atoms | 51 |
| dist_embedding | 64 |
| dist_kappa | 1.0 |
| dist_policy_samples | 8 |
| dist_quantiles | 32 |
| dist_tau_prime_samples | 8 |
| dist_tau_samples | 8 |
| dist_v_max | 110.0 |
| dist_v_min | -10.0 |
| epsilon_anneal_steps | 2000000 |
| epsilon_schedule | linear |
| eval_interval | 1000 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| fork_branches | 1 |
| gradient_clipping | 10.0 |
| graph_eval_episodes | 100 |
| guided_fraction | 0.0 |
| init_from | None |
| initial_collect_steps | 200000 |
| initial_epsilon | 1.0 |
| is_beta | 0.2 |
| is_beta_final | 0.2 |
| is_normalization | batch_max |
| is_weights | True |
| learning_rate | 0.0001 |
| max_steps | 1000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.01 |
| munchausen_alpha | 0.9 |
| munchausen_l0 | -1.0 |
| munchausen_tau | 0.03 |
| n_step_update | 3 |
| priority_exponent | 0.2 |
| rainbow_double | False |
| rainbow_dueling | True |
| rainbow_epsilon_decay | geometric |
| rainbow_epsilon_zero_at | 0.5 |
| rainbow_head | iqn |
| rainbow_munchausen_logpi | online |
| rainbow_noisy | True |
| rainbow_noisy_sigma | 0.5 |
| rainbow_prefill_epsilon | schedule |
| rainbow_stream_width | 512 |
| replay_buffer_max_length | 1048576 |
| replay_ratio | 0.015625 |
| reset_alpha | 0.5 |
| reset_anneal_gamma |  |
| reset_anneal_n_step |  |
| reset_anneal_steps | 10000 |
| reset_interval | 0 |
| reset_stop_after | 0 |
| seed | 4 |
| target_update_period | 500 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b47d-btrpaper-seed4](b47d-btrpaper-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 22.95 | 22.95 | 8.0 | 42.0 | 17.805 | 0.0 | 0.87759 |
| 2000 | 9.07 | 16.01 | 0.0 | 22.0 | 4.771 | 0.0 | 0.85027 |
| 3000 | 7.96 | 13.33 | 0.0 | 17.0 | 3.691 | 0.0 | 0.82381 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 989000 | 94.31 | 94.65 | 31.0 | 95.0 | 190.95 | 98.0 | 0.0 |
| 990000 | 95.0 | 94.66 | 95.0 | 95.0 | 193.643 | 100.0 | 0.0 |
| 991000 | 94.87 | 94.67 | 92.0 | 95.0 | 188.529 | 95.0 | 0.0 |
| 992000 | 94.85 | 94.66 | 87.0 | 95.0 | 189.453 | 96.0 | 0.0 |
| 993000 | 94.95 | 94.66 | 92.0 | 95.0 | 191.599 | 98.0 | 0.0 |
| 994000 | 94.34 | 94.64 | 36.0 | 95.0 | 188.956 | 96.0 | 0.0 |
| 995000 | 94.82 | 94.64 | 87.0 | 95.0 | 188.369 | 95.0 | 0.0 |
| 996000 | 94.93 | 94.64 | 90.0 | 95.0 | 191.521 | 98.0 | 0.0 |
| 997000 | 95.0 | 94.65 | 95.0 | 95.0 | 193.642 | 100.0 | 0.0 |
| 998000 | 94.87 | 94.69 | 84.0 | 95.0 | 191.46 | 98.0 | 0.0 |
| 999000 | 94.68 | 94.71 | 83.0 | 95.0 | 187.093 | 94.0 | 0.0 |
| 1000000 | 94.97 | 94.73 | 92.0 | 95.0 | 192.598 | 99.0 | 0.0 |
