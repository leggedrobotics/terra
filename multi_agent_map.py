import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt


def combine_maps_into_global(map_batch, n_envs_x, n_envs_y):
    """
    Combine a batch of 64x64 maps into a global map (e.g., 2x2 → 128x128).
    map_batch: JAX array or NumPy array of shape [n_envs, 64, 64]
    """
    if hasattr(map_batch, 'device_buffer'):
        map_batch = np.array(map_batch)  # Convert from JAX to NumPy

    maps = map_batch.reshape((n_envs_y, n_envs_x, 64, 64))
    global_map = np.block([[maps[y, x] for x in range(n_envs_x)] for y in range(n_envs_y)])
    print(f"Global map shape: {global_map.shape}")  # Should be (128, 128)
    print(global_map)
    return global_map

def save_global_map_image(global_map, output_path="global_map.png"):
    """
    Save a grayscale image of the global map for visualization or LLM prompt.
    Negative values (targets) become dark; higher values become lighter.
    """
    normalized = (global_map - global_map.min()) / (global_map.max() - global_map.min() + 1e-5)
    img_array = (normalized * 255).astype(np.uint8)
    image = Image.fromarray(img_array)
    image.save(output_path)
    return output_path

    import matplotlib.pyplot as plt

def save_colored_global_map(global_map, output_path="global_map_color.png"):
    plt.imshow(global_map, cmap='viridis')  # or use a custom cmap
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return output_path