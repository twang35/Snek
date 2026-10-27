# b47c-btrpaper-seed3

step **1,000,000** · 1000 evals · trailing **94.66** · peak **94.82** @897,000 · sef **45.3** · best30 **96.7** @898,000

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
| seed | 3 |
| target_update_period | 500 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b47c-btrpaper-seed3](b47c-btrpaper-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 15.83 | 15.83 | 5.0 | 35.0 | 11.442 | 0.0 | 0.8776 |
| 2000 | 3.76 | 9.79 | 0.0 | 18.0 | 1.573 | 0.0 | 0.85028 |
| 3000 | 9.49 | 9.69 | 0.0 | 23.0 | 4.593 | 0.0 | 0.82381 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 989000 | 94.97 | 94.72 | 92.0 | 95.0 | 192.587 | 99.0 | 0.0 |
| 990000 | 94.85 | 94.72 | 89.0 | 95.0 | 189.381 | 96.0 | 0.0 |
| 991000 | 94.84 | 94.72 | 88.0 | 95.0 | 188.448 | 95.0 | 0.0 |
| 992000 | 94.39 | 94.72 | 46.0 | 95.0 | 190.982 | 98.0 | 0.0 |
| 993000 | 94.16 | 94.69 | 16.0 | 95.0 | 190.754 | 98.0 | 0.0 |
| 994000 | 94.74 | 94.69 | 86.0 | 95.0 | 187.234 | 94.0 | 0.0 |
| 995000 | 94.88 | 94.69 | 92.0 | 95.0 | 187.281 | 94.0 | 0.0 |
| 996000 | 94.81 | 94.69 | 88.0 | 95.0 | 188.31 | 95.0 | 0.0 |
| 997000 | 94.53 | 94.68 | 85.0 | 95.0 | 179.736 | 87.0 | 0.0 |
| 998000 | 94.77 | 94.67 | 81.0 | 95.0 | 189.25 | 96.0 | 0.0 |
| 999000 | 94.92 | 94.68 | 88.0 | 95.0 | 191.457 | 98.0 | 0.0 |
| 1000000 | 94.45 | 94.66 | 45.0 | 95.0 | 190.04 | 97.0 | 0.0 |
