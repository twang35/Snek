# b46h-rainbowlocal-seed8

step **19,000** · 19 evals · trailing **60.53** · peak **60.53** @19,000 · sef **0.0** · best30 **0.0** @19,000

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
| seed | 8 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b46h-rainbowlocal-seed8](b46h-rainbowlocal-seed8.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 23.36 | 23.36 | 0.0 | 43.0 | 18.455 | 0.0 | 0.4 |
| 2000 | 22.79 | 23.07 | 6.0 | 44.0 | 17.76 | 0.0 | 0.4 |
| 3000 | 25.71 | 23.95 | 4.0 | 51.0 | 20.669 | 0.0 | 0.025 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 8000 | 61.77 | 38.94 | 1.0 | 87.0 | 58.452 | 0.0 | 0.0125 |
| 9000 | 70.39 | 42.43 | 18.0 | 87.0 | 68.375 | 0.0 | 0.0125 |
| 10000 | 71.51 | 45.34 | 0.0 | 95.0 | 70.926 | 1.0 | 0.0125 |
| 11000 | 72.42 | 47.8 | 0.0 | 89.0 | 70.965 | 0.0 | 0.0125 |
| 12000 | 73.97 | 49.98 | 0.0 | 95.0 | 73.478 | 1.0 | 0.01247 |
| 13000 | 77.39 | 52.09 | 51.0 | 91.0 | 75.997 | 0.0 | 0.01247 |
| 14000 | 77.01 | 53.87 | 0.0 | 95.0 | 77.612 | 2.0 | 0.01245 |
| 15000 | 78.27 | 55.5 | 0.0 | 95.0 | 77.801 | 1.0 | 0.01246 |
| 16000 | 76.94 | 56.84 | 0.0 | 92.0 | 75.488 | 0.0 | 0.01242 |
| 17000 | 79.39 | 58.16 | 0.0 | 95.0 | 80.841 | 3.0 | 0.0124 |
| 18000 | 78.92 | 59.32 | 0.0 | 93.0 | 77.313 | 0.0 | 0.01241 |
| 19000 | 82.38 | 60.53 | 1.0 | 95.0 | 81.828 | 1.0 | 0.01237 |
