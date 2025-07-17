# Core JAX
import jax
import jax.numpy as jnp

# Flax for model and state
from flax import linen as nn

import numpy as np
from typing import Optional, Tuple

def layer_init(scale=1.0):
    return nn.initializers.orthogonal(scale)


class Actor(nn.Module):
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64, kernel_init=layer_init(np.sqrt(2)))(x)
        x = nn.tanh(x)
        x = nn.Dense(64, kernel_init=layer_init(np.sqrt(2)))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.action_dim, kernel_init=layer_init(0.01))(x)
        return x
    
        
        
class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64, kernel_init=layer_init(np.sqrt(2)))(x)
        x = nn.tanh(x)
        x = nn.Dense(64, kernel_init=layer_init(np.sqrt(2)))(x)
        x = nn.tanh(x)
        x = nn.Dense(1, kernel_init=layer_init(1.0))(x)
        return x.squeeze(-1)
            