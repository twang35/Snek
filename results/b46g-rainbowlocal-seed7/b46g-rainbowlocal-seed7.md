# b46g-rainbowlocal-seed7

step **19,000** · 19 evals · trailing **60.74** · peak **60.74** @19,000 · sef **0.0** · best30 **0.0** @19,000

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
| seed | 7 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b46g-rainbowlocal-seed7](b46g-rainbowlocal-seed7.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 16.76 | 16.76 | 0.0 | 39.0 | 11.915 | 0.0 | 0.4 |
| 2000 | 22.55 | 19.66 | 0.0 | 39.0 | 17.565 | 0.0 | 0.4 |
| 3000 | 26.49 | 21.93 | 5.0 | 45.0 | 21.492 | 0.0 | 0.05 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 8000 | 60.76 | 36.7 | 0.0 | 86.0 | 56.678 | 0.0 | 0.0125 |
| 9000 | 67.93 | 40.17 | 30.0 | 84.0 | 64.142 | 0.0 | 0.0125 |
| 10000 | 72.14 | 43.36 | 37.0 | 89.0 | 69.038 | 0.0 | 0.0125 |
| 11000 | 75.18 | 46.26 | 4.0 | 90.0 | 72.824 | 0.0 | 0.0125 |
| 12000 | 76.57 | 48.78 | 0.0 | 90.0 | 74.471 | 0.0 | 0.0125 |
| 13000 | 77.44 | 50.99 | 1.0 | 95.0 | 77.249 | 2.0 | 0.0125 |
| 14000 | 79.1 | 52.99 | 0.0 | 95.0 | 81.006 | 4.0 | 0.0125 |
| 15000 | 81.32 | 54.88 | 26.0 | 95.0 | 85.067 | 6.0 | 0.01246 |
| 16000 | 82.06 | 56.58 | 2.0 | 95.0 | 86.047 | 6.0 | 0.01238 |
| 17000 | 82.79 | 58.12 | 1.0 | 95.0 | 84.013 | 3.0 | 0.01227 |
| 18000 | 82.95 | 59.5 | 32.0 | 95.0 | 84.15 | 3.0 | 0.01218 |
| 19000 | 82.99 | 60.74 | 1.0 | 95.0 | 84.002 | 3.0 | 0.01215 |
