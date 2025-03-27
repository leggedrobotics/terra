import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
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
    local_map_obstacles = np.array(local_map["local_map_obstacles"])

    local_map_obstacles = np.squeeze(local_map_obstacles)


    # Normalize the values for visualization
    #normalized_map = (local_map_obstacles - local_map_obstacles.min()) / (local_map_obstacles.max() - local_map_obstacles.min())

    # Create a heatmap using matplotlib
    plt.figure(figsize=(5, 5))
    #plt.imshow(normalized_map, cmap="viridis", interpolation="nearest")
    plt.imshow(local_map_obstacles, cmap="viridis", interpolation="nearest")

    plt.axis("off")

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg", bbox_inches="tight", pad_inches=0)
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