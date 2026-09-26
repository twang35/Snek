# b46i-rainbowfork25-seed9

step **3,000,000** · 3000 evals · trailing **93.54** · peak **94.46** @1,969,000 · sef **92.0** · best30 **96.6** @924,000

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
| replay_ratio | 0.25 |
| reset_alpha | 0.5 |
| reset_anneal_gamma |  |
| reset_anneal_n_step |  |
| reset_anneal_steps | 10000 |
| reset_interval | 0 |
| reset_stop_after | 0 |
| seed | 9 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

## Resumes

Resumed at 790,000

![b46i-rainbowfork25-seed9](b46i-rainbowfork25-seed9.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 2.92 | 2.92 | 0.0 | 7.0 | 1.996 | 0.0 | 0.4 |
| 2000 | 6.19 | 4.55 | 1.0 | 26.0 | 3.517 | 0.0 | 0.4 |
| 3000 | 12.91 | 7.34 | 2.0 | 29.0 | 9.027 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 93.62 | 93.42 | 16.0 | 95.0 | 185.109 | 93.0 | 0.002 |
| 2990000 | 94.2 | 93.52 | 28.0 | 95.0 | 184.488 | 92.0 | 0.002 |
| 2991000 | 93.13 | 93.49 | 12.0 | 95.0 | 178.577 | 87.0 | 0.002 |
| 2992000 | 94.51 | 93.53 | 62.0 | 95.0 | 185.972 | 93.0 | 0.002 |
| 2993000 | 94.41 | 93.63 | 60.0 | 95.0 | 182.781 | 90.0 | 0.002 |
| 2994000 | 94.85 | 93.65 | 91.0 | 95.0 | 186.221 | 93.0 | 0.002 |
| 2995000 | 94.46 | 93.67 | 64.0 | 95.0 | 178.65 | 86.0 | 0.002 |
| 2996000 | 92.92 | 93.6 | 16.0 | 95.0 | 174.087 | 83.0 | 0.002 |
| 2997000 | 92.91 | 93.58 | 28.0 | 95.0 | 181.307 | 90.0 | 0.002 |
| 2998000 | 93.47 | 93.61 | 12.0 | 95.0 | 178.817 | 87.0 | 0.002 |
| 2999000 | 91.25 | 93.54 | 10.0 | 95.0 | 178.628 | 89.0 | 0.002 |
| 3000000 | 90.6 | 93.54 | 12.0 | 95.0 | 173.931 | 85.0 | 0.002 |
