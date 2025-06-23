from setuptools import find_packages, setup

# Specifying all dependencies, including direct and indirect, for clarity
requires = [
    "jax",
    "jaxlib",
    "chex",
    "tqdm",
    "flax",
    "matplotlib",
    "pygame",
    "wandb",
    "tensorflow_probability",
    "osmnx",
    "opencv-python",
    "scikit-image",
]

setup(
    name="terra",
    version="0.0.1",
    keywords="memory, environment, agent, rl, jax, gym, grid, gridworld, excavator",
    description="Minimalistic grid map environment built with JAX",
    packages=find_packages(),
    install_requires=requires,
    python_requires=">=3.10",
)
