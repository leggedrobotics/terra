# Environment structure

## Background
Terra is a grid world environment, where one or more agents can be trained to modify its terrain conditions and obtain the desired result in terms of height of each of the grid blocks.

## Goal
The goal of the Terra environment is to support the training of RL policies that can potentially be applied in the real world. For this reason, it has to be designed in a way that its dynamics resemble the real world as much as possible.

Ideas that might make the environment more real world-like:
- occupancy layer (binary map indicating the presence of obstacles)
- agents with different size compared to the map tiles
- apply more than one agent action per step (if compatible)
- agent-frame cilindrical coordinates tiles

Other probably out of scope ideas:
- non-rectangular maps
    - possible to obtain the same result by defining obstacles all around in a wider rectangular map
- multi-agent


## Environment
The Terra environment includes:
- maps
- agent states

It takes as input the agent actions, and outputs the updated state of the action map and agent states.

## Maps
The environment includes two maps:
- the target map
    - immutable map used as goal
- the action map
    - mutable map used in the RL loop

The agent is going to interact with the action map, whereas the target map is used to compute the rewards.

Both maps are rectangular 2D elevation maps.
The map is a rectangle, with side length ranging from 8 to 256.
The (0, 0) coordinate is at the top left.

Coordinates increase on the right (X), and down (Y). Angles are measured as zero if the angle vector is aligned with the X axis, and are positive going from X to Y axis.

The maps are fixed-size int32 tensors containing the height information for every tile.

A negative value can be interpreted as the terrain being excavated, and a positive value as the terrain being higher in elevation compared to the zero ground level.

## Agent
The agent has fixed properties as:
- base dimensions (the agent generally has different dimension compared to the tile size)
- arm length (determines the reachability of the tiles, e.g. only tiles at distance < length can be excavated) 

## Agent State
The state of the agent is defined as:
- (X, Y) of the base
    - can be encoded by the cell index that the center of the base is currently occupying
- angle of the base (4 discrete values)
- angle of the cabin (8 discrete values values)

## Actions
The agent actions are defined as:
- base forward
- base backward
- rotate base clockwise
- rotate base anti-clockwise
- rotate cabin clockwise
- rotate cabin anti-clockwise
- do (dig if empty, dump if full)

## Rewards
The reward is given at the end of the episode, if the digged heightmap matches the target heightmap.

Negative rewards are given for:
- digging the wrong tile
- dumping in a digged tile
- moving the base out of the map or on an unaccessible tile (any tile that is not at 0 height level)
- visiting the same state visited before (in terms of map & agent combined state) (?)

On top of this, curriculum learning can be implemented by rewarding:
- a dig action in the right tile
- a dump action in a non-digged tile

## End of the episode
The episode ends if:
- the target map matches the action map
- the episode lasts more than N steps (heuristically computed based on the size of the map and the digging profile)
