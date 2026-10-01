# b48a-rainbowepsgreedy-seed1

step **3,000,000** · 3000 evals · trailing **81.21** · peak **83.54** @1,274,000 · sef **0.0** · best30 **9.1** @1,910,000

## Config

| | |
|---|---|
| adam_epsilon | 0.00015 |
| algo | rainbow |
| batch_size | 32 |
| beta_anneal_steps | 750000 |
| btr_blocks | 3 |
| btr_layer_norm | False |
| btr_residual | False |
| btr_spectral_norm | True |
| collect_envs | 1 |
| discount | 0.99 |
| dist_atoms | 51 |
| dist_embedding | 64 |
| dist_kappa | 1.0 |
| dist_policy_samples | 8 |
| dist_quantiles | 32 |
| dist_tau_prime_samples | 8 |
| dist_tau_samples | 8 |
| dist_v_max | 110.0 |
| dist_v_min | -10.0 |
| epsilon_anneal_steps | 62500 |
| epsilon_schedule | linear |
| eval_interval | 1000 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| fork_branches | 1 |
| gradient_clipping | 0.0 |
| graph_eval_episodes | 100 |
| guided_fraction | 0.0 |
| init_from | None |
| initial_collect_steps | 20000 |
| initial_epsilon | 1.0 |
| is_beta | 0.4 |
| is_beta_final | 1.0 |
| is_normalization | batch_max |
| is_weights | True |
| learning_rate | 6.25e-05 |
| max_steps | 3000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.01 |
| munchausen_alpha | 0.0 |
| munchausen_l0 | -1.0 |
| munchausen_tau | 0.03 |
| n_step_update | 3 |
| priority_exponent | 0.5 |
| rainbow_double | True |
| rainbow_dueling | True |
| rainbow_epsilon_decay | linear |
| rainbow_epsilon_zero_at | 0.0 |
| rainbow_head | c51 |
| rainbow_munchausen_logpi | target |
| rainbow_noisy | False |
| rainbow_noisy_sigma | 0.5 |
| rainbow_prefill_epsilon | random |
| rainbow_stream_width | 512 |
| replay_buffer_max_length | 1000000 |
| replay_ratio | 0.25 |
| reset_alpha | 0.5 |
| reset_anneal_gamma |  |
| reset_anneal_n_step |  |
| reset_anneal_steps | 10000 |
| reset_interval | 0 |
| reset_stop_after | 0 |
| seed | 1 |
| target_update_period | 2000 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b48a-rainbowepsgreedy-seed1](b48a-rainbowepsgreedy-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 6.1 | 4.04 | 0.0 | 17.0 | 4.481 | 0.0 | 0.98418 |
| 2000 | 1.98 | 1.98 | 0.0 | 8.0 | 1.42 | 0.0 | 0.96832 |
| 3000 | 6.78 | 4.95 | 0.0 | 42.0 | 6.057 | 0.0 | 0.9525 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 84.68 | 81.74 | 72.0 | 95.0 | 89.343 | 6.0 | 0.01 |
| 2990000 | 83.2 | 81.71 | 67.0 | 95.0 | 84.829 | 3.0 | 0.01 |
| 2991000 | 83.51 | 81.33 | 70.0 | 95.0 | 83.176 | 1.0 | 0.01 |
| 2992000 | 84.31 | 81.3 | 25.0 | 95.0 | 90.94 | 8.0 | 0.01 |
| 2993000 | 79.23 | 81.37 | 53.0 | 92.0 | 77.875 | 0.0 | 0.01 |
| 2994000 | 80.27 | 81.17 | 57.0 | 95.0 | 83.889 | 5.0 | 0.01 |
| 2995000 | 84.94 | 81.28 | 68.0 | 95.0 | 91.526 | 8.0 | 0.01 |
| 2996000 | 80.24 | 81.19 | 62.0 | 95.0 | 79.84 | 1.0 | 0.01 |
| 2997000 | 83.33 | 81.19 | 64.0 | 95.0 | 87.915 | 6.0 | 0.01 |
| 2998000 | 79.59 | 80.98 | 60.0 | 93.0 | 78.234 | 0.0 | 0.01 |
| 2999000 | 79.51 | 80.98 | 14.0 | 90.0 | 78.129 | 0.0 | 0.01 |
| 3000000 | 81.96 | 81.21 | 58.0 | 93.0 | 80.591 | 0.0 | 0.01 |
