# b49h-btrnospectral-seed8

step **1,000,000** · 1000 evals · trailing **93.44** · peak **93.69** @652,000 · sef **45.3** · best30 **94.1** @997,000

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
| btr_spectral_norm | False |
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
| seed | 8 |
| target_update_period | 500 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b49h-btrnospectral-seed8](b49h-btrnospectral-seed8.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.69 | 1.69 | 0.0 | 16.0 | 0.63 | 0.0 | 0.8776 |
| 2000 | 12.24 | 6.96 | 0.0 | 40.0 | 7.332 | 0.0 | 0.85028 |
| 3000 | 10.47 | 8.13 | 0.0 | 27.0 | 6.098 | 0.0 | 0.82381 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 989000 | 94.68 | 93.0 | 63.0 | 95.0 | 192.312 | 99.0 | 0.0 |
| 990000 | 94.11 | 93.05 | 15.0 | 95.0 | 188.625 | 96.0 | 0.0 |
| 991000 | 92.05 | 93.11 | 11.0 | 95.0 | 177.212 | 87.0 | 0.0 |
| 992000 | 93.91 | 93.15 | 17.0 | 95.0 | 189.466 | 97.0 | 0.0 |
| 993000 | 94.89 | 93.19 | 84.0 | 95.0 | 192.527 | 99.0 | 0.0 |
| 994000 | 94.69 | 93.22 | 64.0 | 95.0 | 192.336 | 99.0 | 0.0 |
| 995000 | 93.19 | 93.32 | 10.0 | 95.0 | 184.629 | 93.0 | 0.0 |
| 996000 | 93.89 | 93.37 | 16.0 | 95.0 | 188.452 | 96.0 | 0.0 |
| 997000 | 94.52 | 93.41 | 75.0 | 95.0 | 187.018 | 94.0 | 0.0 |
| 998000 | 94.62 | 93.42 | 71.0 | 95.0 | 188.217 | 95.0 | 0.0 |
| 999000 | 93.81 | 93.48 | 35.0 | 95.0 | 189.351 | 97.0 | 0.0 |
| 1000000 | 93.62 | 93.44 | 14.0 | 95.0 | 188.171 | 96.0 | 0.0 |
