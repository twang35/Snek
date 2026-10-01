# b49a-btrplaintrunk-seed1

step **879,000** · 879 evals · trailing **94.49** · peak **94.66** @851,000 · sef **34.7** · best30 **95.0** @873,000

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
| seed | 1 |
| target_update_period | 500 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b49a-btrplaintrunk-seed1](b49a-btrplaintrunk-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 11.54 | 11.54 | 0.0 | 27.0 | 8.534 | 0.0 | 0.87759 |
| 2000 | 3.6 | 7.57 | 0.0 | 13.0 | 2.69 | 0.0 | 0.85027 |
| 3000 | 10.04 | 8.39 | 0.0 | 22.0 | 5.591 | 0.0 | 0.82381 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 868000 | 94.36 | 94.57 | 56.0 | 95.0 | 184.655 | 92.0 | 0.0 |
| 869000 | 94.42 | 94.57 | 37.0 | 95.0 | 192.049 | 99.0 | 0.0 |
| 870000 | 93.96 | 94.55 | 6.0 | 95.0 | 186.357 | 94.0 | 0.0 |
| 871000 | 94.91 | 94.57 | 89.0 | 95.0 | 190.407 | 97.0 | 0.0 |
| 872000 | 94.69 | 94.56 | 83.0 | 95.0 | 187.06 | 94.0 | 0.0 |
| 873000 | 93.74 | 94.56 | 4.0 | 95.0 | 184.067 | 92.0 | 0.0 |
| 874000 | 94.67 | 94.55 | 89.0 | 95.0 | 184.924 | 92.0 | 0.0 |
| 875000 | 94.21 | 94.54 | 80.0 | 95.0 | 180.312 | 88.0 | 0.0 |
| 876000 | 94.66 | 94.53 | 76.0 | 95.0 | 188.048 | 95.0 | 0.0 |
| 877000 | 94.8 | 94.53 | 90.0 | 95.0 | 187.156 | 94.0 | 0.0 |
| 878000 | 93.9 | 94.5 | 2.0 | 95.0 | 189.455 | 97.0 | 0.0 |
| 879000 | 94.66 | 94.49 | 87.0 | 95.0 | 184.948 | 92.0 | 0.0 |
