import os
import random
import time
from args import Args

import gymnasium as gym
import numpy as np

# Core JAX
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

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
from agent import Actor, Critic
from tqdm import tqdm

def create_train_state(rng, model, obs_shape, learning_rate, max_grad_norm, num_iterations):
    params = model.init(rng, jnp.ones(obs_shape))

    lr_schedule = optax.linear_schedule(init_value=learning_rate, end_value=0.1 * learning_rate, transition_steps=num_iterations)
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_schedule))
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_env(env_name, idx, capture_video, run_name):
    """
    Creates a thunk (function with no arguments) that initializes a Gym environment with optional video recording and episode statistics recording.
    Args:
        env_name (str): The environment ID to create (as per Gym registry).
        idx (int): The index of the environment (used to determine if video should be captured).
        capture_video (bool): Whether to capture video for the environment (only for idx == 0).
        run_name (str): The name of the run, used to organize video output directories.
    Returns:
        function: A thunk that, when called, returns the initialized Gym environment.
    Note:
        This function is taken from the CleanRL library.
    """
    
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_name, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk
    
if __name__ == "__main__":
    # Create environment and experiment setup
    args = tyro.cli(Args)
    run_name = f"{args.env_name}_{args.exp_name}_{args.seed}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_name, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    
    # calculate batch_size, minibatch_size, num_iterations
    args.batch_size = int(args.num_envs * args.num_steps) # e.g. 4 * 128 = 512
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # e.g. 512 // 4 = 128
    args.num_iterations = args.total_timesteps // args.batch_size # e.g. 500_000 // 512 = 976
    
    # rollout values
    obs = jnp.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape)
    actions = jnp.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
    logprobs = jnp.zeros((args.num_steps, args.num_envs))
    rewards = jnp.zeros((args.num_steps, args.num_envs))
    dones = jnp.zeros((args.num_steps, args.num_envs))
    values = jnp.zeros((args.num_steps, args.num_envs))
    
    
    
    # AGENT STATE
    rng = jax.random.PRNGKey(args.seed)
    rng, critic_key, actor_key = jax.random.split(rng, 3)
    
    critic = Critic()
    critic_state = create_train_state(
        rng=critic_key,
        model=critic,
        obs_shape=jnp.array(envs.single_observation_space.shape),
        learning_rate=args.learning_rate,
        num_iterations=args.num_iterations,
        max_grad_norm=args.max_grad_norm
    )
    
    actor = Actor(action_dim=envs.single_action_space.n)
    actor_state = create_train_state(
        rng=actor_key,
        model=actor,
        obs_shape=jnp.array(envs.single_observation_space.shape),
        learning_rate=args.learning_rate, 
        num_iterations=args.num_iterations, # used for scheduling annealing
        max_grad_norm=args.max_grad_norm
    )
    
    # TRAINING LOOP
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = jnp.zeros(args.num_envs)
    

    @jit
    def value_fn(params, obs):
        value = jax.vmap(
                lambda x: critic_state.apply_fn(params, x))(obs)
        return value

    @jit
    def policy_fn(params, obs):
        return jax.vmap(lambda x: actor_state.apply_fn(params, x))(obs)
    
    @jit
    def actor_loss_fn(params, obs_batch, action, old_logprobs, advantages):
        logits = policy_fn(params, obs_batch)
        log_probs = jax.nn.log_softmax(logits)
        probs = jax.nn.softmax(logits)
        action = action.astype(jnp.int32)
        new_selected_log_prob = jnp.take_along_axis(log_probs, action[:, None], axis=1).squeeze(-1)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        log_ratio = new_selected_log_prob - old_logprobs
        ratio = jnp.exp(log_ratio)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        pg_loss1 = advantages * ratio
        pg_loss2 = advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.minimum(pg_loss1, pg_loss2).mean()
        entropy_loss = entropy.mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        return -(pg_loss - args.ent_coef * entropy_loss), approx_kl  # negative for gradient ascent
    
    @jit
    def critic_loss_fn(params, obs_batch, returns):
        value = value_fn(params, obs_batch)
        v_loss = 0.5 * ((value - returns) ** 2).mean()
        return v_loss
    
    for iteration in range(1, args.num_iterations + 1):
        
        
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs = obs.at[step].set(next_obs)
            dones = dones.at[step].set(next_done)
            
            ## calculate actions and values for each step
            value = value_fn(critic_state.params, next_obs) # calculate state values accross environments
            logits = policy_fn(actor_state.params, next_obs)
            rng, action_key = jax.random.split(rng)
            action = jax.random.categorical(action_key, logits)
            step_log_probs = jax.nn.log_softmax(logits)
            selected_log_prob = jnp.take_along_axis(step_log_probs, action[:, None], axis=1).squeeze(-1)
            
            actions = actions.at[step].set(action)
            logprobs = logprobs.at[step].set(selected_log_prob)
            values = values.at[step].set(value)
            
            
            next_obs, reward, terminations, truncations, infos = envs.step(np.array(action))
            next_done = jnp.logical_or(terminations, truncations)
            rewards = rewards.at[step].set(reward)
            
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        next_value = value_fn(critic_state.params, next_obs)
        advantages = jnp.zeros_like(rewards)
        last_gae_lambda = 0
        
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                mask = 1.0 - next_done
                next_values = next_value
            else:
                mask = 1 - dones[t+1]
                next_values = values[t+1]
            
            td_error =\
                rewards[t] +\
                    args.gamma * next_values * mask -\
                    values[t]
            last_gae_lambda = td_error +\
                args.gamma * args.gae_lambda * mask * last_gae_lambda        
            advantages = advantages.at[t].set(last_gae_lambda)
        returns = advantages + values
        
        
        # flatten rollout values for batching
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = jnp.arange(args.batch_size)
        clip_fracs = [] 
        for epoch in range(args.update_epochs):
            rng, batch_key = jax.random.split(rng, 2)
            b_inds = jax.random.permutation(batch_key, b_inds)
            
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = b_obs[mb_inds]                
                # Compute gradients and update actor and critic
                mb_actions = b_actions[mb_inds]
                mb_logprobs = b_logprobs[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                (actor_loss, approx_kl), actor_grad = value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params, mb_obs, mb_actions, mb_logprobs, mb_advantages)
                critic_loss, critic_grad = value_and_grad(critic_loss_fn)(critic_state.params, mb_obs, mb_returns)
                # Update states
                critic_state = critic_state.apply_gradients(grads=critic_grad)
                actor_state = actor_state.apply_gradients(grads=actor_grad)
            if args.target_kl is not None and approx_kl > args.target_kl:
                break
       
        writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", actor_loss.item(), global_step)
        if iteration == 1:
            pbar = tqdm(total=args.num_iterations, desc="Training Progress")
        pbar.update(1)
        pbar.set_postfix(SPS=int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    pbar.close()     
    envs.close()
    writer.close()    
   
        