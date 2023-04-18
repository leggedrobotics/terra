# Terra
Terra is an open-source platform designed to provide a flexible and abstracted grid world environment for training intelligent agents. Through Terra, researchers and developers can experiment with different reinforcement learning algorithms and techniques, developing new models for controlling complex systems and training agents that can perform tasks that were previously challenging.

Terra is designed with a grid world environment that abstracts away the complexity of the real world, allowing for the development of simplified but realistic simulations of earthmoving equipment such as excavator backhoes, trucks, bulldozers, and more. The game engine provides a rich and diverse set of challenges that allow agents to learn how to perform complex tasks such as digging, hauling, grading, and more.

## Tests
The tests have to be run using the module option, for example:
~~~
python3 -m tests.test_agent
~~~

## Installation
To install the required JAX dependencies, follow [this link](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier).

## Visualization
If you want to render the environment, you are going to need the following Python modules:
~~~
matplotlib
PyQt5
~~~

## Run on GPU
To run JAX on GPU, prepend to the python command the following global variable: `JAX_PLATFORMS="cuda"` or `JAX_PLATFORMS="gpu"` (whichever works). For CPU, `JAX_PLATFORMS="cpu"`.

To check which device you are using, you can:
~~~
print(f"Device = {jnp.ones(1).device_buffer.device()}\n")
~~~
