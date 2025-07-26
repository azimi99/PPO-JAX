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
from flax.training import train_state

# Optax for optimization
import optax

# tensorboard logging module
from torch.utils.tensorboard import SummaryWriter
# import agent
from networks.mlp import Actor, Critic
from tqdm import tqdm


def create_train_state(rng, model, obs_shape, learning_rate, max_grad_norm, num_iterations, decay_lr):
    params = model.init(rng, jnp.ones(obs_shape))
    if decay_lr:
        lr_schedule = optax.linear_schedule(init_value=learning_rate, end_value=0, transition_steps=num_iterations)
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr_schedule))
    else:
        tx = optax.adam(learning_rate)
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)



def train(args, envs, run_name):
    if not run_name: # if not specified
        run_name = f"{args.env_name}_seed_{args.seed}"

    time.sleep(0.1)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    

    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    
    # calculate batch_size, minibatch_size, num_iterations
    args.batch_size = int(args.num_envs * args.num_steps) # e.g. 4 * 128 = 512
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # e.g. 512 // 4 = 128
    args.num_iterations = args.total_timesteps // args.batch_size # e.g. 500_000 // 512 = 976
    
    # rollout values
    obs = np.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape)
    actions = np.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
    logprobs = np.zeros((args.num_steps, args.num_envs))
    rewards = np.zeros((args.num_steps, args.num_envs))
    dones = np.zeros((args.num_steps, args.num_envs))
    values = np.zeros((args.num_steps, args.num_envs))
      
    # AGENT STATE
    rng = jax.random.PRNGKey(args.seed)
    rng, critic_key, actor_key = jax.random.split(rng, 3)
    
    critic = Critic(hidden_layers = (128,128))
    critic_state = create_train_state(
        rng=critic_key,
        model=critic,
        obs_shape=jnp.array(envs.single_observation_space.shape),
        learning_rate=args.learning_rate,
        num_iterations=args.num_iterations,
        max_grad_norm=args.max_grad_norm,
        decay_lr=args.decay_lr
    )
    
    actor = Actor(hidden_layers = (128,128), action_dim=envs.single_action_space.n)
    actor_state = create_train_state(
        rng=actor_key,
        model=actor,
        obs_shape=jnp.array(envs.single_observation_space.shape),
        learning_rate=args.learning_rate, 
        num_iterations=args.num_iterations, # used for scheduling annealing
        max_grad_norm=args.max_grad_norm,
        decay_lr=args.decay_lr
    )
    
    
    def value_fn(params, obs):
        value = jax.vmap(
                lambda x: critic_state.apply_fn(params, x))(obs)
        return value

    
    def policy_fn(params, obs):
        return jax.vmap(lambda x: actor_state.apply_fn(params, x))(obs)
    
    @jit
    def calculate_rollout_statistics(actor_state, critic_state, next_obs, rng):
        value = value_fn(critic_state.params, next_obs) # calculate state values accross environments
        logits = policy_fn(actor_state.params, next_obs)
        rng, action_key = jax.random.split(rng)
        action = jax.random.categorical(action_key, logits)
        step_log_probs = jax.nn.log_softmax(logits)
        selected_log_prob = jnp.take_along_axis(step_log_probs, action[:, None], axis=1).squeeze(-1)
        return action, value, selected_log_prob, rng
       
    @jit
    def train_step(actor_state, critic_state, rollout):
        (mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns) = rollout
            
        def actor_loss_fn(params, obs_batch, action, old_logprobs, advantages):
            logits = policy_fn(params, obs_batch)
            log_probs = jax.nn.log_softmax(logits)
            probs = jax.nn.softmax(logits)
            action = action.astype(jnp.int32)
            new_selected_log_prob = jnp.take_along_axis(log_probs, action[:, None], axis=1).squeeze(-1)
            entropy = -jnp.sum(probs * log_probs, axis=-1)
            log_ratio = new_selected_log_prob - old_logprobs
            ratio = jnp.exp(log_ratio)
            if args.norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            pg_loss1 = advantages * ratio
            pg_loss2 = advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = jnp.minimum(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            return -(pg_loss + args.ent_coef * entropy_loss), approx_kl  # negative for gradient ascent
        
    
        def critic_loss_fn(params, obs_batch, returns):
            value = value_fn(params, obs_batch)
            
            v_loss = ((value - returns) ** 2)
            if args.clip_vloss:
                v_loss_clipped = jnp.clip(v_loss, -args.clip_coef, args.clip_coef)
                v_loss = jnp.maximum(v_loss, v_loss_clipped)
                            
            return 0.5 * v_loss.mean()
        
        (actor_loss, approx_kl), actor_grad =\
            value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params, mb_obs, mb_actions, mb_logprobs, mb_advantages)
        critic_loss, critic_grad = \
            value_and_grad(critic_loss_fn)(critic_state.params, mb_obs, mb_returns)
        # Update states
        critic_state = critic_state.apply_gradients(grads=critic_grad)
        actor_state = actor_state.apply_gradients(grads=actor_grad)
        return actor_state, critic_state, actor_loss, critic_loss, approx_kl
    
    @jit
    def compute_advantages(init_gae_lambda, rollout_values):
        def advantage_step(last_gae_lambda, rollout_values_t):
            (reward, value, next_value, next_done) = rollout_values_t
            td_error =\
                reward +\
                    args.gamma * next_value * (1-next_done) -\
                    value
            last_gae_lambda = td_error +\
                args.gamma * args.gae_lambda * (1-next_done) * last_gae_lambda        
            return last_gae_lambda, last_gae_lambda
        _, advantages = jax.lax.scan(advantage_step, init_gae_lambda, rollout_values, reverse=True)
        return advantages
    
    
    def split_minibatches(data, batch_size, minibatch_size):
        num_minibatches = batch_size // minibatch_size
        return jnp.reshape(data, (num_minibatches, minibatch_size, *data.shape[1:]))
    
    @jit
    def perform_minibatch_updates(init_state, minibatches):
        def minibatch_step(carry, minibatch):
            actor_state, critic_state = carry
            mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns = minibatch
            actor_state, critic_state, actor_loss, critic_loss, approx_kl = train_step(
                actor_state, critic_state, 
                (mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns)
            )
            return (actor_state, critic_state), (actor_loss, critic_loss, approx_kl)
        (actor_state, critic_state), (actor_loss, critic_loss, approx_kl) \
            = jax.lax.scan(minibatch_step, init_state, minibatches)
        return actor_state, critic_state, actor_loss[-1], critic_loss[-1], approx_kl[-1]
       
        
    # TRAINING LOOP
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = jnp.zeros(args.num_envs)
    uploade_video_file = []    
    
    for iteration in range(1, args.num_iterations + 1):
        
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            ## calculate actions and values for each step
            action, value, selected_log_prob, rng = \
                calculate_rollout_statistics(actor_state, critic_state, next_obs, rng=rng)
            
            actions[step] = action
            logprobs[step] = selected_log_prob
            values[step] = value
            
            
            next_obs, reward, terminations, truncations, infos = envs.step(np.array(action))
            next_done = jnp.logical_or(terminations, truncations)
            rewards[step] = reward

            if "episode" in infos:
                writer.add_scalar("charts/episodic_return", np.array(infos["episode"]["r"]).mean(), global_step)
                writer.add_scalar("charts/episodic_length", np.array(infos["episode"]["l"]).mean(), global_step)
        next_value = value_fn(critic_state.params, next_obs)
        advantages = np.zeros_like(rewards)
        
        ## Compute GAE
        next_values = np.concatenate([values[1:], next_value[None, :]], axis=0)
        next_dones = np.concatenate([dones[1:], next_done[None, :]], axis=0)

        rollout_values = (rewards, values, next_values, next_dones)
        init_gae_lambda = jnp.zeros_like(rewards[0])

        jnp_advantages = compute_advantages(init_gae_lambda, rollout_values)
        advantages = np.asarray(jnp_advantages) 
        
        returns = advantages + values
        
        # flatten rollout values for batching
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = jnp.arange(args.batch_size)
        
        
        for epoch in range(args.update_epochs):
            rng, batch_key = jax.random.split(rng, 2)
            b_inds = jax.random.permutation(batch_key, b_inds)
            
            # split batch of data into minibatches
            mb_obs = split_minibatches(b_obs[b_inds], args.batch_size, args.minibatch_size)
            mb_actions = split_minibatches(b_actions[b_inds], args.batch_size, args.minibatch_size)
            mb_logprobs = split_minibatches(b_logprobs[b_inds], args.batch_size, args.minibatch_size)
            mb_advantages = split_minibatches(b_advantages[b_inds], args.batch_size, args.minibatch_size)
            mb_returns = split_minibatches(b_returns[b_inds], args.batch_size, args.minibatch_size)
            minibatches = (mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns)
            # train over minibatches
            init_state = (actor_state, critic_state)
            actor_state, critic_state, actor_loss, critic_loss, approx_kl =\
                perform_minibatch_updates(init_state, minibatches=minibatches)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
       
        writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", actor_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        if iteration == 1:
            pbar = tqdm(total=args.num_iterations, desc="Training Progress")
        pbar.update(1)
        pbar.set_postfix(SPS=int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if args.capture_video and args.track:
            video_dir = f"videos/{run_name}"
            if os.path.exists(video_dir):
                video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
                for video_file in video_files:
                    video_path = os.path.join(video_dir, video_file)
                    if video_path not in uploade_video_file:
                        wandb.log({"video": wandb.Video(video_path, format="mp4")})
                        uploade_video_file.append(video_path)
        
    pbar.close()     
    envs.close()
    if args.track:
        wandb.finish()
    writer.close()
    return actor_state, critic_state    
    
   
        