# Environment structure

## Background
Terra is a grid world environment, where one or more agents can be trained to modify its terrain conditions and obtain the desired result in terms of height of each of the grid blocks.

## Goal
The goal of the Terra environment is to support the training of RL policies that can be applied in the real world.

Ideas that might make the environment more real world-like:
- agent-frame cilindrical coordinates tiles
- non-rectangular maps

## Environment
The Terra environment includes:
- maps
- agent states

It takes as input the agent actions, and outputs the updated state of the action map and agent states.

## Maps
The environment includes multiple maps:
- the target map
    - immutable height map used as goal (-X for did, +1 for dumpable tile)
    - type: (NxM) int32 tensor
- the action map
    - mutable height map used in the RL loop
    - type: (NxM) int32 tensor
- the traversability mask map
    - mutable map to indicate if the robot can or can't traverse a given tile
    - type: (NxM) int32 tensor (only 0 and 1 allowed)

The maps are rectangular 2D maps. The side length is ranging from 8 to 128.
The (0, 0) coordinate is at the top left.

Coordinates increase on the right (X), and down (Y). Angles are measured as zero if the angle vector is aligned with the X axis, and are positive going from X to Y axis.

In the height maps, a negative value can be interpreted as the terrain being excavated, and a positive value as the terrain being higher in elevation compared to the zero ground level.

In the mask maps, 1 means dumpable/traversable.

## Agent
The agent has fixed properties as:
- X base dimension (in number of tiles)
- Y base dimension (in number of tiles)
- arm length (in number of tiles)
    - determines which tiles are reachable

Regulating the robot's dimensions with respect to the tile size in the map, will have the effect of scaling up/down the granularity of the map. This way we can go from one map tile being of the same size as the robot's shovel, to one tile being e.g. 5x the shovel.

## Agent State
We define two types of agents:
- wheeled
- tracked

The state space of the agents is the same, but their action space differs.

The state of the agent is defined as:
- (X, Y) of the geometric center of the base
- angle of the base (4 discrete values)
- angle of the cabin ([2X_base + 2Y_base + 4*(2*E -1)] discrete values)
    - where E>=1 is the length of the arm

## Observation
The system is fully observable, so the observation is the current state of the maps and the agent state.

Alternative observations (as first person view), will be implemented as a wrapper around the environment, allowing a case-by-case use.

## Actions
The *wheeled* agent actions are defined as:
- base forward
- base backward
- rotate base clockwise + forward
- rotate base clockwise + backward
- rotate base anti-clockwise + forward
- rotate base anti-clockwise + backward
- rotate cabin clockwise
- rotate cabin anti-clockwise
- extend arm
- retract arm
- do (dig if empty, dump if full)

The *tracked* agent actions are defined as:
- base forward
- base backward
- rotate base clockwise
- rotate base anti-clockwise
- rotate cabin clockwise
- rotate cabin anti-clockwise
- extend arm
- retract arm
- do (dig if empty, dump if full)

## Rewards
The reward is given at the end of the episode, if the digged heightmap matches the target heightmap (negative values), and if all the positive values lie on the allowed dumpable target area.

On top of the end of episode reward, we can implement curriculum learning by adding the following intermediate rewards.

Negative rewards are given for:
- using any action that is not "do"
- existing
- digging the wrong tile
- dumping where it is not allowed to dump
- traversing a tile forbidden by the traversability map
- moving the base on an unaccessible tile (any tile that is not at 0 height level)
- dumping in a digged tile (?)

Positive rewards are given for:
- a dig action in one of the right tiles
- a dump action in an allowed tile

## End of the episode
The episode ends if:
- the target map matches the action map (negative values only)
- the episode lasts more than N steps (heuristically computed based on the size of the map and the digging profile)


# Integration in the real system
This paragraph briefly explains how the RL policy (built with this environment interfaces) is intended to be integrated with the pre-existing system.

The RL policy takes as input:
- the maps
- the current agent state

and gives as output:
- the next agent action

The RL policy is going to be run online on the robot, to be able to incorporate information available online incoming from the perception stack.

The policy is run iteratively in open-loop until an actionable task (a task defined by the "do" action) can be performed. Between two "do" tasks, the RL policy runs open-loop, followed by no action from the low-level planners.

## Locomotion
Let's consider an example. The policy might output several subsequent "forward" actions with 0.5m station increments. At the end of this series of actions, there is a "do" action. At this point, the RRT* planner comes into place, planning from point A (initial pose) to B (pose where the do action occurs) directly.\
The RRT* existing planner is currently informed about traversable areas by an occupancy map - so we can offload the obstacle avoidance part of the complexity to RRT*.

## Dig and Dump
Also in the case of digging and dumping, the cabin controller and arm planner are only engaged once a "do" action is reached from the previous one.

For example, the robot digs (do), then the RL policy outputs three subsequent cabin rotations, and then a dump action (do) is used again. Only at this point, the lower-level systems are called.

## Converting actions into states
In order to implement such a system based on open-loop segments, a model capable of integrating RL policy actions into robot states (at least on the high level in terms of occupancy grids) is required.\
This model needs to use the same logic implemented in the Terra training environment.

# Implementation
Terra will be GPU-accelerated and based on Pytorch 2.

TODO
