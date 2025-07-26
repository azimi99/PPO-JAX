
import os
from dataclasses import dataclass
@dataclass
# This Args structure is taken from the CleanRL library (https://github.com/vwxyzjn/cleanrl)
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")] # experiment name
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "ppo jax"
    wandb_entity: str = None # wandb team
    capture_video: bool = False # render environment frames

    """Algorithm specific arguments"""
    env_name: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 3e-4
    num_envs: int = 1 # parrallel envs
    num_steps: int = 128 # rollout length
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.01 # entropy coefficient
    vf_coef: float = 0.5 # value function coefficient
    max_grad_norm: float = 0.5 # used for gradient clipping
    target_kl: float = None # when set stop training if threshold passed
    decay_lr: bool = False # toggle learning rate schedule

    """Runtime computed variables"""
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0