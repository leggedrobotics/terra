from setuptools import find_packages
from setuptools import setup

requires = ["jax", "jaxlib", "chex"]

setup(
    name="terra",
    version="0.0.1",
    keywords="memory, environment, agent, rl, jax, gym, grid, gridworld, excavator",
    description="Minimalistic grid map environment built with JAX",
    packages=find_packages(),
    install_requires=requires,
    python_requires=">=3.11",
)
