# b46n-rainbowfork0625-seed14

step **3,000,000** · 3000 evals · trailing **92.84** · peak **93.86** @1,237,000 · sef **66.8** · best30 **94.5** @1,237,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | rainbow |
| batch_size | 32 |
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
| replay_ratio | 0.0625 |
| reset_alpha | 0.5 |
| reset_anneal_gamma |  |
| reset_anneal_n_step |  |
| reset_anneal_steps | 10000 |
| reset_interval | 0 |
| reset_stop_after | 0 |
| seed | 14 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b46n-rainbowfork0625-seed14](b46n-rainbowfork0625-seed14.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.23 | 0.23 | 0.0 | 2.0 | -1.035 | 0.0 | 0.4 |
| 2000 | 5.99 | 3.11 | 0.0 | 18.0 | 3.015 | 0.0 | 0.4 |
| 3000 | 7.03 | 4.42 | 0.0 | 37.0 | 3.291 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 91.63 | 92.85 | 6.0 | 95.0 | 183.25 | 93.0 | 0.002 |
| 2990000 | 93.31 | 92.8 | 35.0 | 95.0 | 181.895 | 90.0 | 0.002 |
| 2991000 | 93.49 | 92.79 | 41.0 | 95.0 | 186.131 | 94.0 | 0.002 |
| 2992000 | 93.7 | 92.77 | 35.0 | 95.0 | 188.309 | 96.0 | 0.002 |
| 2993000 | 93.47 | 92.86 | 26.0 | 95.0 | 180.044 | 88.0 | 0.002 |
| 2994000 | 94.33 | 92.95 | 47.0 | 95.0 | 188.922 | 96.0 | 0.002 |
| 2995000 | 94.01 | 92.91 | 24.0 | 95.0 | 186.612 | 94.0 | 0.002 |
| 2996000 | 94.12 | 92.86 | 50.0 | 95.0 | 188.733 | 96.0 | 0.002 |
| 2997000 | 93.98 | 93.0 | 35.0 | 95.0 | 186.608 | 94.0 | 0.002 |
| 2998000 | 91.9 | 92.92 | 25.0 | 95.0 | 180.56 | 90.0 | 0.002 |
| 2999000 | 93.58 | 93.0 | 36.0 | 95.0 | 186.202 | 94.0 | 0.002 |
| 3000000 | 91.22 | 92.84 | 41.0 | 95.0 | 175.848 | 86.0 | 0.002 |
