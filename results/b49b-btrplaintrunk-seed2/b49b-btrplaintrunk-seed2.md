# b49b-btrplaintrunk-seed2

step **1,000,000** · 1000 evals · trailing **94.51** · peak **94.6** @991,000 · sef **41.5** · best30 **95.0** @613,000

## Config

| | |
|---|---|
| adam_epsilon | 1.953125e-05 |
| algo | btr |
| batch_size | 256 |
| beta_anneal_steps | 1 |
| btr_blocks | 3 |
| btr_layer_norm | False |
| btr_residual | False |
| btr_spectral_norm | True |
| collect_envs | 64 |
| discount | 0.997 |
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
| epsilon_schedule | linear |
| eval_interval | 1000 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| fork_branches | 1 |
| gradient_clipping | 10.0 |
| graph_eval_episodes | 100 |
| guided_fraction | 0.0 |
| init_from | None |
| initial_collect_steps | 200000 |
| initial_epsilon | 1.0 |
| is_beta | 0.2 |
| is_beta_final | 0.2 |
| is_normalization | batch_max |
| is_weights | True |
| learning_rate | 0.0001 |
| max_steps | 1000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.01 |
| munchausen_alpha | 0.9 |
| munchausen_l0 | -1.0 |
| munchausen_tau | 0.03 |
| n_step_update | 3 |
| priority_exponent | 0.2 |
| rainbow_double | False |
| rainbow_dueling | True |
| rainbow_epsilon_decay | geometric |
| rainbow_epsilon_zero_at | 0.5 |
| rainbow_head | iqn |
| rainbow_munchausen_logpi | online |
| rainbow_noisy | True |
| rainbow_noisy_sigma | 0.5 |
| rainbow_prefill_epsilon | schedule |
| rainbow_stream_width | 512 |
| replay_buffer_max_length | 1048576 |
| replay_ratio | 0.015625 |
| reset_alpha | 0.5 |
| reset_anneal_gamma |  |
| reset_anneal_n_step |  |
| reset_anneal_steps | 10000 |
| reset_interval | 0 |
| reset_stop_after | 0 |
| seed | 2 |
| target_update_period | 500 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b49b-btrplaintrunk-seed2](b49b-btrplaintrunk-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 5.28 | 5.28 | 0.0 | 30.0 | 4.222 | 0.0 | 0.8776 |
| 2000 | 14.01 | 9.64 | 1.0 | 22.0 | 9.052 | 0.0 | 0.85028 |
| 3000 | 11.78 | 10.36 | 2.0 | 26.0 | 6.716 | 0.0 | 0.82381 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 989000 | 94.75 | 94.57 | 85.0 | 95.0 | 188.16 | 95.0 | 0.0 |
| 990000 | 94.92 | 94.59 | 92.0 | 95.0 | 190.415 | 97.0 | 0.0 |
| 991000 | 94.93 | 94.6 | 88.0 | 95.0 | 192.496 | 99.0 | 0.0 |
| 992000 | 93.8 | 94.57 | 38.0 | 95.0 | 182.044 | 90.0 | 0.0 |
| 993000 | 94.79 | 94.59 | 86.0 | 95.0 | 187.158 | 94.0 | 0.0 |
| 994000 | 93.94 | 94.57 | 10.0 | 95.0 | 187.397 | 95.0 | 0.0 |
| 995000 | 93.69 | 94.53 | 2.0 | 95.0 | 185.074 | 93.0 | 0.0 |
| 996000 | 94.76 | 94.53 | 81.0 | 95.0 | 190.234 | 97.0 | 0.0 |
| 997000 | 94.48 | 94.52 | 84.0 | 95.0 | 185.793 | 93.0 | 0.0 |
| 998000 | 94.5 | 94.52 | 58.0 | 95.0 | 185.808 | 93.0 | 0.0 |
| 999000 | 94.68 | 94.51 | 86.0 | 95.0 | 188.077 | 95.0 | 0.0 |
| 1000000 | 94.69 | 94.51 | 82.0 | 95.0 | 184.956 | 92.0 | 0.0 |
