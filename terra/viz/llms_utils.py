import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import jax
import jax.numpy as jnp
import pygame as pg

def generate_local_map(timestep):
    """
    Generate a local map from the environment's current state.

    Args:
        env: The TerraEnvBatch environment instance.
        timestep: The current timestep of the environment.

    Returns:
        A dictionary representing the local map.
    """
    # Access the state object
    state = timestep.state

    # Access the world (GridWorld) object from the state
    world = state.world
    # Extract relevant local maps from the world
    local_map = {
        "local_map_action_neg": world.local_map_action_neg.map.tolist(),
        "local_map_action_pos": world.local_map_action_pos.map.tolist(),
        "local_map_target_neg": world.local_map_target_neg.map.tolist(),
        "local_map_target_pos": world.local_map_target_pos.map.tolist(),
        "local_map_dumpability": world.local_map_dumpability.map.tolist(),
        "local_map_obstacles": world.local_map_obstacles.map.tolist(),
    }


    return local_map

def local_map_to_image(local_map):
    """
    Convert a local map dictionary to an image.

    Args:
        local_map: A dictionary containing local map data.

    Returns:
        An image (numpy array) representing the local map.
    """
    # Example: Visualize the traversability mask
    local_map_target_pos = np.array(local_map["local_map_target_pos"])

    #local_map_target_pos = np.squeeze(local_map_target_pos)


    # Normalize the values for visualization
    normalized_map = (local_map_target_pos - local_map_target_pos.min()) / (local_map_target_pos.max() - local_map_target_pos.min() + 1e-6)

    # Create a heatmap using matplotlib
    plt.figure(figsize=(5, 5))
    plt.imshow(normalized_map, cmap="viridis", interpolation="nearest")
    #plt.imshow(local_map_target_pos, cmap="viridis", interpolation="nearest")

    plt.axis("off")

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg", bbox_inches="tight", pad_inches=0)
    #plt.savefig("local_map.jpg", format="jpg", bbox_inches="tight", pad_inches=0)
    buf.seek(0)

    # Convert the buffer to a PIL image and then to a numpy array
    img = Image.open(buf)
    img_array = np.array(img)
    buf.close()

    return img_array

def capture_screen(surface):
    """Captures the current screen and converts it to an image format."""
    img_array = pg.surfarray.array3d(surface)
    #img_array = np.rot90(img_array, k=3)  # Rotate if needed
    img_array = np.transpose(img_array, (1, 0, 2))  # Correct rotation

    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_array

def save_video(frames, output_path, fps=1):
    """Saves a list of frames as a video."""
    if len(frames) == 0:
        print("No frames to save.")
        return
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")

def extract_bucket_status(state):
    """
    Extract the bucket status from the state.

    Args:
        state: The current State object.

    Returns:
        str: The bucket status ('loaded' or 'empty').
    """
    # Access the bucket status from the agent's state
    bucket_status = state.agent.agent_state.loaded

    # Map the status to a human-readable string
    return "loaded" if bucket_status else "empty"

def base_orientation_to_direction(angle_base):
    """
    Convert the base orientation value (0-3) to a cardinal direction.

    Args:
        angle_base (int or JAX array): The base orientation value.

    Returns:
        str: The corresponding cardinal direction ('up', 'right', 'down', 'left').
    """
    # Convert JAX array to a Python scalar if necessary
    if isinstance(angle_base, jax.Array):
        angle_base = angle_base.item()

    # Map orientation to cardinal direction
    direction_map = {
        0: "right",
        1: "up",
        2: "left",
        3: "down"
    }
    return direction_map.get(angle_base, "unknown")  # Default to 'unknown' if invalid

def extract_base_orientation(state):
    """
    Extract the excavator's base orientation from the state and convert it to a cardinal direction.

    Args:
        state: The current State object.

    Returns:
        A dictionary containing the base angle and its corresponding cardinal direction.
    """
    # Extract the base angle
    angle_base = state.agent.agent_state.angle_base

    # Convert the base angle to a cardinal direction
    direction = base_orientation_to_direction(angle_base)

    return {
        "angle_base": angle_base,
        "direction": direction,
    }    

def summarize_local_map(local_map):
    """
    Generate a textual summary of the local map.

    Args:
        local_map: A dictionary representing the local map.

    Returns:
        str: A textual summary of the local map.
    """
    num_obstacles = np.sum(np.array(local_map["local_map_obstacles"]) > 0)
    num_dumpable = np.sum(np.array(local_map["local_map_dumpability"]) > 0)
    num_target_pos = np.sum(np.array(local_map["local_map_target_pos"]) > 0)
    num_target_neg = np.sum(np.array(local_map["local_map_target_neg"]) > 0)

    return (
        f"The local map contains {num_obstacles} obstacles, "
        f"{num_dumpable} dumpable areas, "
        f"{num_target_pos} positive target areas, and "
        f"{num_target_neg} negative target areas."
    )

def extract_positions(state):
    """
    Extract the current base position and target position from the game state.

    Args:
        state: The current game state object.

    Returns:
        A tuple containing:
        - current_position: A dictionary with the current base position (x, y).
        - target_position: A dictionary with the target position (x, y), or None if not available.
        
    """

    #print(state.agent.agent_state.pos_base[0])
    # Extract th11e current base position
    current_position = {
        "x": state.agent.agent_state.pos_base[0][0],
        "y": state.agent.agent_state.pos_base[0][1]
    }

    # Extract the target position from the target_map if available
    #print(state.world.target_map.map)
    target_positions = []

    for x in range(state.world.target_map.map.shape[1]):  # Iterate over rows
        for y in range(state.world.target_map.map.shape[2]):  # Iterate over columns
            if state.world.target_map.map[0, x, y] == -1:  # Access the value at (0, x, y)
                target_positions.append((x, y))
    
    # # Convert positions to tuples
    start = (int(current_position["x"]), int(current_position["y"]))
    #target = (int(target_position["x"]), int(target_position["y"])) if target_position else None

    return start, target_positions

def path_to_actions(path, initial_orientation, step_size=1):
    """
    Convert a path into a list of action numbers based on the current base orientation and step size.

    Args:
        path (list of tuples): The path as a list of (x, y) positions.
        initial_orientation (str): The initial base orientation ('up', 'down', 'left', 'right').
        step_size (int): The number of pixels the base moves forward in one step.

    Returns:
        list of int: A list of action numbers corresponding to the path.
    """
    # Define the mapping of directions to deltas
    direction_deltas = {
        "up": (-1, 0),
        "right": (0, 1),
        "down": (1, 0),
        "left": (0, -1),
    }

    # Define the order of directions for turning (90-degree increments)
    directions = ["up", "right", "down", "left"]

    # Define action numbers
    FORWARD = 0
    CLOCK = 2
    ANTICLOCK = 3

    # Helper function to determine the direction between two points
    def get_direction(from_pos, to_pos):
        delta = (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
        # Normalize the delta to match one of the predefined directions
        if delta[0] != 0:
            delta = (delta[0] // abs(delta[0]), 0)
        elif delta[1] != 0:
            delta = (0, delta[1] // abs(delta[1]))
        for direction, d in direction_deltas.items():
            if delta == d:
                return direction
        return None

    # Initialize the list of actions
    actions = []
    current_orientation = initial_orientation

    # Iterate through the simplified path
    for i in range(len(path) - 1):
        current_pos = path[i]
        next_pos = path[i + 1]

        # Determine the required direction to move
        required_direction = get_direction(current_pos, next_pos)
        if required_direction is None:
            raise ValueError(f"Invalid direction between {current_pos} and {next_pos}")

        # Determine the turns needed to face the required direction
        while current_orientation != required_direction:
            current_idx = directions.index(current_orientation)
            required_idx = directions.index(required_direction)

            # Determine if we need to turn right (CLOCK) or left (ANTICLOCK)
            if (required_idx - current_idx) % 4 == 1:  # Clockwise
                actions.append(CLOCK)
                current_orientation = directions[(current_idx + 1) % 4]
            else:  # Counter-clockwise
                actions.append(ANTICLOCK)
                current_orientation = directions[(current_idx - 1) % 4]

        # Add a single forward action for the entire straight-line segment
        actions.append(FORWARD)

    return actions

def find_nearest_target(start, target_positions):
    """
    Find the nearest target position to the starting point.

    Args:
        start (tuple): The starting position as (x, y).
        target_positions (list of tuples): A list of target positions as (x, y).

    Returns:
        tuple: The nearest target position as (x, y), or None if the list is empty.
    """
    if not target_positions:
        return None

    # Calculate the Euclidean distance to each target and find the nearest one
    nearest_target = min(target_positions, key=lambda target: (target[0] - start[0])**2 + (target[1] - start[1])**2)
    return nearest_target