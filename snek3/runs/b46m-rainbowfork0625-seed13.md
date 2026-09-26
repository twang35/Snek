# b46m-rainbowfork0625-seed13

step **3,000,000** · 3000 evals · trailing **92.59** · peak **93.7** @1,435,000 · sef **64.7** · best30 **93.8** @1,439,000

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
| seed | 13 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b46m-rainbowfork0625-seed13](b46m-rainbowfork0625-seed13.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.53 | 0.53 | 0.0 | 1.0 | -3.188 | 0.0 | 0.4 |
| 2000 | 3.93 | 2.23 | 0.0 | 10.0 | 0.104 | 0.0 | 0.4 |
| 3000 | 2.8 | 2.42 | 0.0 | 14.0 | 1.816 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 90.33 | 92.52 | 28.0 | 95.0 | 176.976 | 88.0 | 0.002 |
| 2990000 | 92.81 | 92.63 | 25.0 | 95.0 | 184.469 | 93.0 | 0.002 |
| 2991000 | 91.95 | 92.61 | 17.0 | 95.0 | 183.64 | 93.0 | 0.002 |
| 2992000 | 90.63 | 92.46 | 3.0 | 95.0 | 177.333 | 88.0 | 0.002 |
| 2993000 | 91.45 | 92.47 | 19.0 | 95.0 | 180.093 | 90.0 | 0.002 |
| 2994000 | 92.43 | 92.41 | 22.0 | 95.0 | 180.055 | 89.0 | 0.002 |
| 2995000 | 94.49 | 92.62 | 80.0 | 95.0 | 189.157 | 96.0 | 0.002 |
| 2996000 | 93.83 | 92.54 | 41.0 | 95.0 | 187.485 | 95.0 | 0.002 |
| 2997000 | 92.55 | 92.6 | 37.0 | 95.0 | 182.236 | 91.0 | 0.002 |
| 2998000 | 91.75 | 92.61 | 21.0 | 95.0 | 180.448 | 90.0 | 0.002 |
| 2999000 | 94.16 | 92.62 | 78.0 | 95.0 | 185.834 | 93.0 | 0.002 |
| 3000000 | 92.04 | 92.59 | 31.0 | 95.0 | 178.725 | 88.0 | 0.002 |
