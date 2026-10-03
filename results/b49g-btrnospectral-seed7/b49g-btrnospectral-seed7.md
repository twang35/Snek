# b49g-btrnospectral-seed7

step **1,000,000** · 1000 evals · trailing **93.19** · peak **93.98** @937,000 · sef **44.9** · best30 **96.0** @933,000

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
| seed | 7 |
| target_update_period | 500 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b49g-btrnospectral-seed7](b49g-btrnospectral-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 5.14 | 5.14 | 0.0 | 15.0 | 3.064 | 0.0 | 0.8776 |
| 2000 | 9.03 | 7.08 | 0.0 | 28.0 | 4.705 | 0.0 | 0.85028 |
| 3000 | 8.98 | 7.72 | 0.0 | 23.0 | 4.531 | 0.0 | 0.82382 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 989000 | 93.23 | 92.53 | 13.0 | 95.0 | 185.795 | 94.0 | 0.0 |
| 990000 | 93.83 | 92.5 | 1.0 | 95.0 | 188.393 | 96.0 | 0.0 |
| 991000 | 94.96 | 92.63 | 93.0 | 95.0 | 191.583 | 98.0 | 0.0 |
| 992000 | 93.66 | 92.8 | 52.0 | 95.0 | 185.091 | 93.0 | 0.0 |
| 993000 | 94.43 | 92.84 | 63.0 | 95.0 | 188.974 | 96.0 | 0.0 |
| 994000 | 93.65 | 92.85 | 45.0 | 95.0 | 187.199 | 95.0 | 0.0 |
| 995000 | 92.02 | 92.92 | 46.0 | 95.0 | 176.291 | 86.0 | 0.0 |
| 996000 | 94.61 | 93.01 | 66.0 | 95.0 | 189.148 | 96.0 | 0.0 |
| 997000 | 93.27 | 93.04 | 10.0 | 95.0 | 186.889 | 95.0 | 0.0 |
| 998000 | 92.57 | 93.2 | 10.0 | 95.0 | 185.05 | 94.0 | 0.0 |
| 999000 | 94.88 | 93.22 | 92.0 | 95.0 | 189.553 | 96.0 | 0.0 |
| 1000000 | 92.16 | 93.19 | 12.0 | 95.0 | 176.253 | 86.0 | 0.0 |
