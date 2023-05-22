# Terra
Terra is an open-source platform designed to provide a flexible and abstracted grid world environment for training intelligent agents. Through Terra, researchers and developers can experiment with different reinforcement learning algorithms and techniques, developing new models for controlling complex systems and training agents that can perform tasks that were previously challenging.

Terra is designed with a grid world environment that abstracts away the complexity of the real world, allowing for the development of simplified but realistic simulations of earthmoving equipment such as excavator backhoes, trucks, bulldozers, and more. The game engine provides a rich and diverse set of challenges that allow agents to learn how to perform complex tasks such as digging, hauling, grading, and more.

## Installation
This repo is built on JAX. However, JAX has different versions depending on the hardware you plan to use.

Follow [this link](https://github.com/google/jax#installation) to install the right one for you.

## Run on GPU
To run JAX on GPU, prepend to the python command the following global variable: `JAX_PLATFORMS="cuda"` or `JAX_PLATFORMS="gpu"` (whichever works). For CPU, `JAX_PLATFORMS="cpu"`.

To check which device you are using, you can:
~~~
print(f"Device = {jnp.ones(1).device_buffer.device()}\n")
~~~

## Tests
The tests have to be run using the module option, for example:
~~~
python3 -m tests.test_agent
~~~

## Visualization
If you want to render the environment, you are going to need the following Python modules:
~~~
matplotlib
PyQt5
~~~

## Pre-commit
To setup the development tools, run:
~~~
pip install -r requirements_dev.txt
~~~

Then run:
~~~
pre-commit install
~~~

At this point you should be able to run:
~~~
pre-commit run --all-files
~~~
to check if all files pass the linters and if the unit tests don't fail.

From now on, the `pre-commit` tool will run automatically at every commit.
