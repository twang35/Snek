# b47e-btrlocal-seed5

step **919,000** · 919 evals · trailing **94.4** · peak **94.44** @915,000 · sef **87.9** · best30 **96.6** @903,000

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
| seed | 5 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b47e-btrlocal-seed5](b47e-btrlocal-seed5.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 1.16 | 1.16 | 1.0 | 3.0 | 0.202 | 0.0 | 0.4 |
| 2000 | 2.16 | 1.66 | 1.0 | 4.0 | -0.492 | 0.0 | 0.4 |
| 3000 | 9.61 | 4.31 | 2.0 | 28.0 | 4.603 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 908000 | 93.12 | 94.31 | 21.0 | 95.0 | 186.734 | 95.0 | 0.002 |
| 909000 | 94.61 | 94.34 | 82.0 | 95.0 | 186.207 | 93.0 | 0.002 |
| 910000 | 93.95 | 94.36 | 38.0 | 95.0 | 185.552 | 93.0 | 0.002 |
| 911000 | 94.49 | 94.37 | 59.0 | 95.0 | 185.981 | 93.0 | 0.002 |
| 912000 | 93.72 | 94.33 | 12.0 | 95.0 | 187.382 | 95.0 | 0.002 |
| 913000 | 94.87 | 94.37 | 86.0 | 95.0 | 191.455 | 98.0 | 0.002 |
| 914000 | 94.8 | 94.38 | 76.0 | 95.0 | 191.401 | 98.0 | 0.002 |
| 915000 | 94.99 | 94.44 | 94.0 | 95.0 | 192.583 | 99.0 | 0.002 |
| 916000 | 94.37 | 94.43 | 32.0 | 95.0 | 192.013 | 99.0 | 0.002 |
| 917000 | 94.47 | 94.42 | 48.0 | 95.0 | 191.06 | 98.0 | 0.002 |
| 918000 | 93.56 | 94.37 | 16.0 | 95.0 | 187.222 | 95.0 | 0.002 |
| 919000 | 94.79 | 94.4 | 84.0 | 95.0 | 190.427 | 97.0 | 0.002 |
