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

## Use dataset loaded from disk
Datasets loaded from disk are the main option Terra offers. To load them, define the following env variables when you launch the script.
Note that `DATASET_SIZE` needs to be the same for all the map types you define in the config. So for example, if all the datasets you select have 1300 maps each but one has 1000, then you need to set `DATASET_SIZE=1000` to be able to load all of them (this way you will load only 1000/1300 elements of the other datasets).
~~~
DATASET_PATH=path_to_dataset DATASET_SIZE=dataset_size
~~~

## Environment details
### Maps
Terra relies on multiple maps per environment instance to support multiple functionalities and usecases.

In every environment the following maps are defined:
- target map
    - 1: must dump here to terminate the episode
    - 0: free
    - -1: must dig here 
- action map
    - -1: dug here during the episode
    - 0: free
    - greater than 0: dumped here
- dig map (same as action map but updated on the dig action & before the dump action is complete)
- dumpability mask
    - 1: can dump
    - 0: can't dump
- padding mask
    - 0: traversable
    - 1: non traversable
- traversability mask
    - -1: agent occupancy
    - 0: traversable
    - 1: non traversable
- local map target positive (contains the sum of all the positive target map tiles in a given workspace)
- local map target negative (contains the sum of all the negative target map tiles in a given workspace)
- local map action positive (contains the sum of all the positive action map tiles in a given workspace)
- local map action negative (contains the sum of all the negative action map tiles in a given workspace)
- local obstacles map (contains the sum of all the padding mask tiles in a given workspace)
- local dumpability mask (contains the sum of all the dumpability mask tiles in a given workspace)

### Maps of different size
Note: the environment supports maps of different sizes run on parallel environments at the same time.
However, every map is padded to the right and to the bottom to reach the biggest map size defined in the config.
The agent transition in these parts of the map is automatically blocked by the environment itself.

## Visualization
Terra supports two rendering engines: one based on numpy and the other on pygame.
The former is used for development (manual mode is implemented, so you can control the agent with the keyboard),
but it is very slow at rendering multiple environments at the same time.
The latter is used for nice renderings but doesn't support interactiveness yet.

You can control which one to use at the time of instantiation of the `TerraEnvBatch` class, where you can select
`numpy` or `pygame` as rendering engine.

## Run manual mode
To run in manual mode:
~~~
DATASET_PATH="/path/to/dataset/folders" DATASET_SIZE=1000 python -m viz.main_manual
~~~
For example:
~~~
DATASET_PATH="/home/antonio/img_generator" DATASET_SIZE=1000 python -m viz_pygame.main_manual
~~~

This commands loads DATASET_SIZE maps defined in the `BatchConfig` in `config.py`, from DATASET_PATH (being the folder containing all the maps folders defined in the config).
Then, one is sampled and you can play with it using the keyboard.

Mind that the maps are padded to the max dimensions defined in `ImmutableMapsConfig` in `config.py` - and you can't have maps bigger than that value.
Also, in the config you can set `move_tiles` to select how many tiles per move actions are traversed, and `tile_size` to control how big the agent is compared to the map. The reach of the agent arm is automatically adjusted.

You can generate maps from the `digbench` repo (for simple rectangles you can use `generate_rectangles.py`).

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
