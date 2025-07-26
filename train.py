from algorithms.ppo_jax import train
from util import make_env
from args import Args
import tyro
import os
import shutil
import gymnasium as gym

args = tyro.cli(Args)
run_name = f"{args.env_name}_seed_{args.seed}"
if os.path.exists(f"runs/{run_name}"):
    shutil.rmtree(f"runs/{run_name}")
if os.path.exists(f"videos/{run_name}"):
    shutil.rmtree(f"videos/{run_name}")
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_name, i, args.capture_video, run_name) for i in range(args.num_envs)],
)
_,_ = train(args, envs, run_name=run_name) 