# b48c-rainbowepsgreedy-seed3

step **3,000,000** · 3000 evals · trailing **86.02** · peak **87.53** @2,277,000 · sef **0.0** · best30 **21.4** @2,684,000

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
| seed | 3 |
| target_update_period | 2000 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b48c-rainbowepsgreedy-seed3](b48c-rainbowepsgreedy-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.48 | 0.48 | 0.0 | 3.0 | -0.073 | 0.0 | 0.98418 |
| 2000 | 7.46 | 3.97 | 0.0 | 24.0 | 6.17 | 0.0 | 0.96837 |
| 3000 | 17.14 | 8.36 | 2.0 | 39.0 | 12.682 | 0.0 | 0.95251 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 86.19 | 86.16 | 53.0 | 95.0 | 103.821 | 19.0 | 0.01 |
| 2990000 | 81.35 | 85.87 | 44.0 | 95.0 | 85.969 | 6.0 | 0.01 |
| 2991000 | 84.79 | 86.01 | 65.0 | 95.0 | 99.316 | 16.0 | 0.01 |
| 2992000 | 87.53 | 86.11 | 74.0 | 95.0 | 100.05 | 14.0 | 0.01 |
| 2993000 | 86.54 | 86.25 | 62.0 | 95.0 | 101.06 | 16.0 | 0.01 |
| 2994000 | 85.74 | 86.19 | 62.0 | 95.0 | 93.262 | 9.0 | 0.01 |
| 2995000 | 82.93 | 86.08 | 68.0 | 95.0 | 82.447 | 1.0 | 0.01 |
| 2996000 | 89.53 | 86.4 | 74.0 | 95.0 | 121.045 | 33.0 | 0.01 |
| 2997000 | 87.84 | 85.98 | 57.0 | 95.0 | 107.404 | 21.0 | 0.01 |
| 2998000 | 86.89 | 85.87 | 30.0 | 95.0 | 107.386 | 22.0 | 0.01 |
| 2999000 | 88.98 | 85.92 | 7.0 | 95.0 | 122.558 | 35.0 | 0.01 |
| 3000000 | 86.7 | 86.02 | 54.0 | 95.0 | 97.184 | 12.0 | 0.01 |
