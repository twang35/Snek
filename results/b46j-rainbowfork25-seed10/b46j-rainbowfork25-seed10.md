# b46j-rainbowfork25-seed10

step **3,000,000** · 3000 evals · trailing **93.5** · peak **94.46** @2,183,000 · sef **92.9** · best30 **96.8** @1,288,000

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
| seed | 10 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

## Resumes

Resumed at 770,000

![b46j-rainbowfork25-seed10](b46j-rainbowfork25-seed10.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 13.06 | 13.06 | 0.0 | 57.0 | 11.935 | 0.0 | 0.4 |
| 2000 | 16.34 | 14.7 | 0.0 | 49.0 | 13.541 | 0.0 | 0.4 |
| 3000 | 20.32 | 16.57 | 0.0 | 47.0 | 15.629 | 0.0 | 0.4 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 94.87 | 93.71 | 90.0 | 95.0 | 189.484 | 96.0 | 0.002 |
| 2990000 | 91.3 | 93.72 | 16.0 | 95.0 | 178.744 | 89.0 | 0.002 |
| 2991000 | 92.91 | 93.72 | 18.0 | 95.0 | 184.411 | 93.0 | 0.002 |
| 2992000 | 91.98 | 93.77 | 12.0 | 95.0 | 185.497 | 95.0 | 0.002 |
| 2993000 | 94.28 | 93.81 | 48.0 | 95.0 | 185.862 | 93.0 | 0.002 |
| 2994000 | 92.16 | 93.76 | 14.0 | 95.0 | 185.639 | 95.0 | 0.002 |
| 2995000 | 94.15 | 93.76 | 16.0 | 95.0 | 189.75 | 97.0 | 0.002 |
| 2996000 | 90.82 | 93.66 | 8.0 | 95.0 | 177.18 | 88.0 | 0.002 |
| 2997000 | 90.66 | 93.58 | 18.0 | 95.0 | 176.065 | 87.0 | 0.002 |
| 2998000 | 93.25 | 93.57 | 11.0 | 95.0 | 184.744 | 93.0 | 0.002 |
| 2999000 | 92.72 | 93.57 | 12.0 | 95.0 | 183.272 | 92.0 | 0.002 |
| 3000000 | 92.59 | 93.5 | 16.0 | 95.0 | 182.158 | 91.0 | 0.002 |
