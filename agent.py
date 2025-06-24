# Core JAX
import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad, vmap, lax

# Flax for model and state
from flax import linen as nn
from flax.training import train_state

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

class Agent(nn.Module):
    action_dim: int
    
    def setup(self):
        self.actor = Actor(action_dim=self.action_dim)
        self.critic = Critic()
        
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(
        self, x, rng: jax.Array, action: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        logits = self.actor(x)
        action = jax.random.categorical(rng, logits)
        log_probs = jax.nn.log_softmax(logits)
        selected_log_prob = jnp.take_along_axis(log_probs, action[:, None], axis=1).squeeze(-1)
        
        # compute entropy
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        value = self.critic(x)
        
        return action, selected_log_prob, entropy, value
    
        
        
            