import jax
import jax.numpy as jnp

class GaussianNoise:
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def sample(self, key: jax.Array, shape: tuple) -> jnp.ndarray:
        # generate matrix/vector with entries sampled from 0 mean, sigma^2 variance normal distribution
        return self.sigma * jax.random.normal(key, shape=shape)

    def __call__(self, key: jax.Array, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.sample(key, x.shape)
