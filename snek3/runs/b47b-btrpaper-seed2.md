# b47b-btrpaper-seed2

step **1,000,000** · 1000 evals · trailing **94.13** · peak **94.7** @815,000 · sef **45.0** · best30 **96.7** @871,000

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
| seed | 2 |
| target_update_period | 500 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b47b-btrpaper-seed2](b47b-btrpaper-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 22.93 | 22.93 | 5.0 | 40.0 | 17.801 | 0.0 | 0.8776 |
| 2000 | 1.27 | 12.1 | 0.0 | 15.0 | 0.439 | 0.0 | 0.85028 |
| 3000 | 7.86 | 10.69 | 0.0 | 23.0 | 4.43 | 0.0 | 0.82381 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 989000 | 94.04 | 93.93 | 21.0 | 95.0 | 184.529 | 92.0 | 0.0 |
| 990000 | 94.95 | 93.94 | 90.0 | 95.0 | 192.599 | 99.0 | 0.0 |
| 991000 | 93.65 | 93.99 | 27.0 | 95.0 | 188.114 | 96.0 | 0.0 |
| 992000 | 93.13 | 93.99 | 23.0 | 95.0 | 182.586 | 91.0 | 0.0 |
| 993000 | 94.61 | 94.01 | 59.0 | 95.0 | 191.183 | 98.0 | 0.0 |
| 994000 | 93.84 | 94.06 | 35.0 | 95.0 | 188.4 | 96.0 | 0.0 |
| 995000 | 92.5 | 94.0 | 32.0 | 95.0 | 180.77 | 90.0 | 0.0 |
| 996000 | 94.39 | 94.03 | 43.0 | 95.0 | 188.95 | 96.0 | 0.0 |
| 997000 | 94.24 | 94.06 | 52.0 | 95.0 | 185.798 | 93.0 | 0.0 |
| 998000 | 94.89 | 94.12 | 92.0 | 95.0 | 189.496 | 96.0 | 0.0 |
| 999000 | 94.97 | 94.12 | 92.0 | 95.0 | 192.602 | 99.0 | 0.0 |
| 1000000 | 94.65 | 94.13 | 86.0 | 95.0 | 186.11 | 93.0 | 0.0 |
