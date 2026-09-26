# b47f-btrlocal-seed6

step **919,000** · 919 evals · trailing **94.41** · peak **94.54** @499,000 · sef **87.3** · best30 **96.3** @726,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | btr |
| batch_size | 128 |
| beta_anneal_steps | 300000 |
| btr_blocks | 3 |
| btr_layer_norm | False |
| btr_residual | True |
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
| epsilon_anneal_steps | 2000000 |
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
| max_steps | 1000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.002 |
| munchausen_alpha | 0.9 |
| munchausen_l0 | -1.0 |
| munchausen_tau | 0.03 |
| n_step_update | 3 |
| priority_exponent | 0.6 |
| rainbow_double | False |
| rainbow_dueling | True |
| rainbow_epsilon_decay | geometric |
| rainbow_epsilon_zero_at | 0.0 |
| rainbow_head | quantile |
| rainbow_munchausen_logpi | online |
| rainbow_noisy | True |
| rainbow_noisy_sigma | 0.5 |
| rainbow_prefill_epsilon | schedule |
| rainbow_stream_width | 512 |
| replay_buffer_max_length | 100000 |
| replay_ratio | 1.0 |
| reset_alpha | 0.5 |
| reset_anneal_gamma |  |
| reset_anneal_n_step |  |
| reset_anneal_steps | 10000 |
| reset_interval | 0 |
| reset_stop_after | 0 |
| seed | 6 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b47f-btrlocal-seed6](b47f-btrlocal-seed6.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.41 | 1.41 | 0.0 | 3.0 | 0.144 | 0.0 | 0.4 |
| 2000 | 1.55 | 1.48 | 0.0 | 3.0 | 0.237 | 0.0 | 0.4 |
| 3000 | 2.95 | 1.97 | 1.0 | 6.0 | -0.997 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 908000 | 94.8 | 94.33 | 80.0 | 95.0 | 190.39 | 97.0 | 0.002 |
| 909000 | 94.29 | 94.35 | 47.0 | 95.0 | 186.828 | 94.0 | 0.002 |
| 910000 | 94.25 | 94.33 | 51.0 | 95.0 | 187.786 | 95.0 | 0.002 |
| 911000 | 94.79 | 94.33 | 88.0 | 95.0 | 187.318 | 94.0 | 0.002 |
| 912000 | 94.16 | 94.34 | 31.0 | 95.0 | 186.711 | 94.0 | 0.002 |
| 913000 | 93.78 | 94.35 | 37.0 | 95.0 | 186.354 | 94.0 | 0.002 |
| 914000 | 93.46 | 94.3 | 24.0 | 95.0 | 187.088 | 95.0 | 0.002 |
| 915000 | 94.76 | 94.3 | 82.0 | 95.0 | 189.332 | 96.0 | 0.002 |
| 916000 | 94.76 | 94.32 | 88.0 | 95.0 | 188.339 | 95.0 | 0.002 |
| 917000 | 94.4 | 94.39 | 59.0 | 95.0 | 190.012 | 97.0 | 0.002 |
| 918000 | 94.57 | 94.39 | 68.0 | 95.0 | 188.099 | 95.0 | 0.002 |
| 919000 | 94.85 | 94.41 | 89.0 | 95.0 | 189.426 | 96.0 | 0.002 |
