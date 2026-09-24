# b48b-rainbowepsgreedy-seed2

step **3,000,000** · 3000 evals · trailing **85.81** · peak **88.38** @2,648,000 · sef **0.0** · best30 **19.9** @2,654,000

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
| seed | 2 |
| target_update_period | 2000 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b48b-rainbowepsgreedy-seed2](b48b-rainbowepsgreedy-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.9 | 1.9 | 0.0 | 13.0 | 1.295 | 0.0 | 0.98421 |
| 2000 | 1.33 | 1.61 | 0.0 | 8.0 | 0.775 | 0.0 | 0.96837 |
| 3000 | 1.23 | 1.49 | 0.0 | 8.0 | 0.669 | 0.0 | 0.95253 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 86.78 | 85.71 | 68.0 | 95.0 | 97.3 | 12.0 | 0.01 |
| 2990000 | 85.38 | 85.81 | 62.0 | 95.0 | 89.922 | 6.0 | 0.01 |
| 2991000 | 88.44 | 85.71 | 20.0 | 95.0 | 108.936 | 22.0 | 0.01 |
| 2992000 | 87.25 | 85.35 | 4.0 | 95.0 | 100.769 | 15.0 | 0.01 |
| 2993000 | 86.94 | 85.5 | 56.0 | 95.0 | 96.493 | 11.0 | 0.01 |
| 2994000 | 86.2 | 85.58 | 42.0 | 95.0 | 100.721 | 16.0 | 0.01 |
| 2995000 | 86.2 | 85.53 | 68.0 | 95.0 | 94.712 | 10.0 | 0.01 |
| 2996000 | 88.42 | 85.74 | 72.0 | 95.0 | 109.937 | 23.0 | 0.01 |
| 2997000 | 87.62 | 85.73 | 60.0 | 95.0 | 101.131 | 15.0 | 0.01 |
| 2998000 | 83.6 | 85.68 | 50.0 | 95.0 | 84.175 | 2.0 | 0.01 |
| 2999000 | 87.09 | 85.77 | 54.0 | 95.0 | 97.696 | 12.0 | 0.01 |
| 3000000 | 84.1 | 85.81 | 14.0 | 95.0 | 90.629 | 8.0 | 0.01 |
