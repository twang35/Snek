# b49f-btrnospectral-seed6

step **1,000,000** · 1000 evals · trailing **92.8** · peak **93.85** @617,000 · sef **46.8** · best30 **94.9** @908,000

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
| seed | 6 |
| target_update_period | 500 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b49f-btrnospectral-seed6](b49f-btrnospectral-seed6.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 9.46 | 9.46 | 0.0 | 23.0 | 5.302 | 0.0 | 0.8776 |
| 2000 | 6.64 | 8.05 | 0.0 | 18.0 | 2.443 | 0.0 | 0.85028 |
| 3000 | 8.95 | 8.35 | 1.0 | 22.0 | 4.198 | 0.0 | 0.82381 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 989000 | 91.76 | 92.85 | 10.0 | 95.0 | 173.809 | 84.0 | 0.0 |
| 990000 | 93.41 | 92.95 | 20.0 | 95.0 | 187.92 | 96.0 | 0.0 |
| 991000 | 94.98 | 93.0 | 93.0 | 95.0 | 192.624 | 99.0 | 0.0 |
| 992000 | 92.71 | 93.09 | 56.0 | 95.0 | 177.993 | 87.0 | 0.0 |
| 993000 | 92.96 | 93.07 | 13.0 | 95.0 | 187.484 | 96.0 | 0.0 |
| 994000 | 91.46 | 93.03 | 26.0 | 95.0 | 172.531 | 83.0 | 0.0 |
| 995000 | 90.22 | 93.0 | 16.0 | 95.0 | 165.081 | 77.0 | 0.0 |
| 996000 | 93.23 | 92.94 | 16.0 | 95.0 | 180.563 | 89.0 | 0.0 |
| 997000 | 94.56 | 92.94 | 78.0 | 95.0 | 187.064 | 94.0 | 0.0 |
| 998000 | 91.59 | 92.85 | 8.0 | 95.0 | 183.039 | 93.0 | 0.0 |
| 999000 | 93.98 | 92.82 | 12.0 | 95.0 | 188.588 | 96.0 | 0.0 |
| 1000000 | 94.19 | 92.8 | 49.0 | 95.0 | 189.752 | 97.0 | 0.0 |
