# b46e-rainbowlocal-seed5

step **19,000** · 19 evals · trailing **43.95** · peak **43.95** @19,000 · sef **0.0** · best30 **0.0** @19,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | rainbow |
| batch_size | 128 |
| beta_anneal_steps | 300000 |
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
| epsilon_anneal_steps | 1 |
| epsilon_schedule | eval |
| eval_interval | 1000 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| fork_branches | 4 |
| fork_max_steps | 60 |
| fork_min_length | 85 |
| fork_prob | 0.5 |
| gradient_clipping | 0.0 |
| graph_eval_episodes | 100 |
| guided_fraction | 0.8 |
| init_from | None |
| initial_collect_steps | 2000 |
| initial_epsilon | 0.4 |
| is_beta | 0.4 |
| is_beta_final | 1.0 |
| is_normalization | mean |
| is_weights | True |
| learning_rate | 1e-05 |
| max_steps | 3000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.002 |
| munchausen_alpha | 0.0 |
| munchausen_l0 | -1.0 |
| munchausen_tau | 0.03 |
| n_step_update | 3 |
| priority_exponent | 0.6 |
| rainbow_double | True |
| rainbow_dueling | True |
| rainbow_epsilon_decay | linear |
| rainbow_epsilon_zero_at | 0.0 |
| rainbow_head | c51 |
| rainbow_munchausen_logpi | target |
| rainbow_noisy | True |
| rainbow_noisy_sigma | 0.5 |
| rainbow_prefill_epsilon | random |
| rainbow_stream_width | 512 |
| replay_buffer_max_length | 100000 |
| replay_ratio | 1.0 |
| reset_alpha | 0.5 |
| reset_anneal_gamma |  |
| reset_anneal_n_step |  |
| reset_anneal_steps | 10000 |
| reset_interval | 0 |
| reset_stop_after | 0 |
| seed | 5 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b46e-rainbowlocal-seed5](b46e-rainbowlocal-seed5.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 13.32 | 13.32 | 1.0 | 37.0 | 8.731 | 0.0 | 0.4 |
| 2000 | 20.15 | 16.73 | 1.0 | 38.0 | 15.302 | 0.0 | 0.4 |
| 3000 | 20.45 | 17.97 | 2.0 | 33.0 | 15.472 | 0.0 | 0.1 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 8000 | 31.21 | 23.98 | 8.0 | 57.0 | 26.137 | 0.0 | 0.025 |
| 9000 | 35.49 | 25.26 | 2.0 | 55.0 | 30.431 | 0.0 | 0.0125 |
| 10000 | 40.9 | 26.82 | 8.0 | 70.0 | 35.973 | 0.0 | 0.0125 |
| 11000 | 47.83 | 28.73 | 19.0 | 85.0 | 42.742 | 0.0 | 0.0125 |
| 12000 | 47.81 | 30.32 | 14.0 | 79.0 | 43.008 | 0.0 | 0.0125 |
| 13000 | 55.21 | 32.24 | 14.0 | 88.0 | 50.858 | 0.0 | 0.0125 |
| 14000 | 63.33 | 34.46 | 28.0 | 91.0 | 59.841 | 0.0 | 0.0125 |
| 15000 | 63.13 | 36.37 | 17.0 | 88.0 | 59.525 | 0.0 | 0.0125 |
| 16000 | 66.63 | 38.26 | 17.0 | 89.0 | 63.9 | 0.0 | 0.0125 |
| 17000 | 72.07 | 40.25 | 4.0 | 95.0 | 70.893 | 1.0 | 0.0125 |
| 18000 | 74.25 | 42.14 | 25.0 | 90.0 | 72.017 | 0.0 | 0.0125 |
| 19000 | 76.53 | 43.95 | 46.0 | 90.0 | 74.756 | 0.0 | 0.01248 |
