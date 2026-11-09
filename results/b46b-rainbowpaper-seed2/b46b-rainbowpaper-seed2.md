# b46b-rainbowpaper-seed2

step **3,000,000** · 3000 evals · trailing **93.86** · peak **94.76** @1,271,000 · sef **67.1** · best30 **97.5** @1,700,000

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
| seed | 2 |
| target_update_period | 2000 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b46b-rainbowpaper-seed2](b46b-rainbowpaper-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.59 | 0.59 | 0.0 | 4.0 | 0.036 | 0.0 | 0.0 |
| 2000 | 0.57 | 0.58 | 0.0 | 3.0 | 0.016 | 0.0 | 0.0 |
| 3000 | 2.95 | 1.37 | 0.0 | 46.0 | 2.298 | 0.0 | 0.0 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 93.89 | 93.97 | 30.0 | 95.0 | 185.37 | 93.0 | 0.0 |
| 2990000 | 94.64 | 93.97 | 77.0 | 95.0 | 189.168 | 96.0 | 0.0 |
| 2991000 | 94.73 | 94.03 | 75.0 | 95.0 | 189.279 | 96.0 | 0.0 |
| 2992000 | 93.98 | 94.01 | 29.0 | 95.0 | 188.523 | 96.0 | 0.0 |
| 2993000 | 93.47 | 94.01 | 18.0 | 95.0 | 184.812 | 93.0 | 0.0 |
| 2994000 | 94.14 | 94.06 | 30.0 | 95.0 | 188.704 | 96.0 | 0.0 |
| 2995000 | 92.77 | 94.06 | 17.0 | 95.0 | 188.329 | 97.0 | 0.0 |
| 2996000 | 94.36 | 94.06 | 56.0 | 95.0 | 186.903 | 94.0 | 0.0 |
| 2997000 | 93.39 | 94.05 | 29.0 | 95.0 | 185.842 | 94.0 | 0.0 |
| 2998000 | 94.19 | 93.99 | 33.0 | 95.0 | 187.672 | 95.0 | 0.0 |
| 2999000 | 93.53 | 93.94 | 35.0 | 95.0 | 184.875 | 93.0 | 0.0 |
| 3000000 | 92.44 | 93.86 | 12.0 | 95.0 | 179.756 | 89.0 | 0.0 |
