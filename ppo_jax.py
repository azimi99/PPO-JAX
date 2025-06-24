import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

# Core JAX
import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad, vmap, lax

# Flax for model and state
from flax import linen as nn
from flax.training import train_state

# Optax for optimization
import optax
# command line tool
import tyro
# tensorboard logging module
from torch.utils.tensorboard import SummaryWriter
# import agent
from agent import Agent


def make_env(env_id, idx, capture_video, run_name):
    """
    Creates a thunk (function with no arguments) that initializes a Gym environment with optional video recording and episode statistics recording.
    Args:
        env_id (str): The environment ID to create (as per Gym registry).
        idx (int): The index of the environment (used to determine if video should be captured).
        capture_video (bool): Whether to capture video for the environment (only for idx == 0).
        run_name (str): The name of the run, used to organize video output directories.
    Returns:
        function: A thunk that, when called, returns the initialized Gym environment.
    Note:
        This function is adapted from the CleanRL library.
    """
    
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    

if __name__ == "__main__":
    # Create environment and experiment setup
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    

    
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    
    # calculate batch_size, minibatch_size, num_iterations
    args.batch_size = int(args.num_envs * args.num_steps) # e.g. 4 * 128 = 512
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # e.g. 512 // 4 = 128
    args.num_iterations = args.total_timesteps // args.batch_size # e.g. 500_000 // 512 = 976
    
    # Algorithm storage

    obs = jnp.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape)
    actions = jnp.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
    logprobs = jnp.zeros((args.num_steps, args.num_envs))
    rewards = jnp.zeros((args.num_steps, args.num_envs))
    dones = jnp.zeros((args.num_steps, args.num_envs))
    values = jnp.zeros((args.num_steps, args.num_envs))
    

    # TRAINING LOOP
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = jnp.zeros(args.num_envs)
    
    # AGENT STATE
    rng = jax.random.PRNGKey(args.seed)
    # Calculate the product of the tuple values for action_dim
    action_dim = int(np.prod(envs.single_action_space.shape))
    agent = Agent(action_dim=action_dim)
    params = agent.init(
        rng,
        jnp.zeros((1,) + envs.single_observation_space.shape),
        rng=rng,
        method=Agent.get_action_and_value
    )
    rng, step_rng = jax.random.split(rng)
    env_rngs = jax.random.split(step_rng, args.num_envs)

    # vmap over batch dimension (axis 0)
    batched_get_action_and_value = jax.vmap(
        lambda obs, rng: agent.apply(params, obs[None, ...], rng, method=Agent.get_action_and_value),
        in_axes=(0, 0)
    )
    actions, logprobs, entropies, values = batched_get_action_and_value(next_obs, env_rngs)

    for iteration in range(args.num_iterations):

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs = obs.at[step].set(next_obs)
            dones = dones.at[step].set(next_done)
             
        # calculate objectives
        
        for t in reversed(range(args.num_steps)):
            pass
        
        
        # Policy Optimization Loop
        
        for epoch in range(args.update_epochs):
            # ...
            for start in range(0, args.batch_size, args.minibatch_size):
                pass
        
    
        