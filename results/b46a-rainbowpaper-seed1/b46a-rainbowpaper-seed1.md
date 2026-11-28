# b46a-rainbowpaper-seed1

step **3,000,000** · 3000 evals · trailing **94.19** · peak **94.75** @2,142,000 · sef **73.4** · best30 **98.1** @2,270,000

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
| epsilon_anneal_steps | 1 |
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
| initial_epsilon | 0.0 |
| is_beta | 0.4 |
| is_beta_final | 1.0 |
| is_normalization | batch_max |
| is_weights | True |
| learning_rate | 6.25e-05 |
| max_steps | 3000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.0 |
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
| rainbow_noisy | True |
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

![b46a-rainbowpaper-seed1](b46a-rainbowpaper-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 8.81 | 8.81 | 0.0 | 18.0 | 4.634 | 0.0 | 0.0 |
| 2000 | 0.67 | 4.74 | 0.0 | 4.0 | 0.116 | 0.0 | 0.0 |
| 3000 | 0.48 | 3.32 | 0.0 | 3.0 | -0.074 | 0.0 | 0.0 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 94.72 | 94.29 | 67.0 | 95.0 | 192.33 | 99.0 | 0.0 |
| 2990000 | 93.98 | 94.26 | 64.0 | 95.0 | 183.514 | 91.0 | 0.0 |
| 2991000 | 93.92 | 94.23 | 41.0 | 95.0 | 183.324 | 91.0 | 0.0 |
| 2992000 | 93.39 | 94.14 | 8.0 | 95.0 | 184.924 | 93.0 | 0.0 |
| 2993000 | 94.97 | 94.18 | 92.0 | 95.0 | 192.633 | 99.0 | 0.0 |
| 2994000 | 93.77 | 94.16 | 49.0 | 95.0 | 187.368 | 95.0 | 0.0 |
| 2995000 | 94.96 | 94.2 | 93.0 | 95.0 | 191.628 | 98.0 | 0.0 |
| 2996000 | 94.05 | 94.15 | 29.0 | 95.0 | 185.607 | 93.0 | 0.0 |
| 2997000 | 93.97 | 94.18 | 48.0 | 95.0 | 188.551 | 96.0 | 0.0 |
| 2998000 | 94.92 | 94.21 | 92.0 | 95.0 | 189.497 | 96.0 | 0.0 |
| 2999000 | 94.44 | 94.19 | 65.0 | 95.0 | 186.93 | 94.0 | 0.0 |
| 3000000 | 94.23 | 94.19 | 51.0 | 95.0 | 187.795 | 95.0 | 0.0 |
