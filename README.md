## PPO in JAX
This is a gymnasium compatible implementation of PPO in JAX. 
Here are some Gymnasium environments with continuous observation spaces and discrete action spaces suitable for benchmarking this PPO JAX implementation:

- `CartPole-v1`
- `MountainCar-v0`
- `Acrobot-v1`
- `LunarLander-v3`

## Key Features
| Feature                         | Description                  | Algorithm                   |
|----------------------------------|------------------------------|------------------------------|
| Action Space Compatibility       | Discrete | `ppo_jax.py` |
| Environment Space Compatibility  | Continuous |  |
| Gym Support                     | ✅ | |
| Cuda Support                     | Coming Soon | |

## Results
Check out our experiment results and interactive visualizations below:

[Experiment Report & Visualizations](https://wandb.ai/azimi/ppo%20jax/reports/Results-from-PPO-JAX--VmlldzoxMzY3OTMxNg?accessToken=otlcvfgm3a53jfkwmh0owf3cxwe79603airegk60uxlol0ecl93mb3vb35kn0rps))

## Technical Writeup


A detailed technical write-up for this project is available here:  
[PPO-JAX Blog Post](https://www.fromscratchdev.io/rl-blog/ppo-jax-post.html)

## References

- **Proximal Policy Optimization Algorithms**  
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).  
    [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

- **CleanRL: High-quality single-file implementations of RL algorithms**  
    [CleanRL GitHub Repository](https://github.com/vwxyzjn/cleanrl)
