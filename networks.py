# Core JAX
import jax
import jax.numpy as jnp

# Flax for model and state
from flax import linen as nn
from dataclasses import field

import numpy as np
from typing import Tuple, Callable

def layer_init(scale=1.0):
    return nn.initializers.orthogonal(scale)


class Actor(nn.Module):
    action_dim: int
    hidden_layers: Tuple[int,...] = field(default_factory=lambda: (64, 64))
    activation_fn: Callable = field(nn.tanh)
    
    @nn.compact
    def __call__(self, x):
        for hidden_size in self.hidden_layers:
            x = nn.Dense(hidden_size, kernel_init=layer_init(np.sqrt(2)))(x)
            x = self.activation_fn(x)
        x = nn.Dense(self.action_dim, kernel_init=layer_init(0.01))(x)
        return x
    
        
        
class Critic(nn.Module):
    hidden_layers: Tuple[int,...] = field(default_factory=lambda: (64, 64))
    activation_fn: Callable = field(nn.tanh)
    
    @nn.compact
    def __call__(self, x):
        for hidden_size in self.hidden_layers:
            x = nn.Dense(hidden_size, kernel_init=layer_init(np.sqrt(2)))(x)
            x = self.activation_fn(x)
        x = nn.Dense(1, kernel_init=layer_init(1.0))(x)
        return x.squeeze(-1)
            