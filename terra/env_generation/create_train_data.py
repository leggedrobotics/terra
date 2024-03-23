import os
import yaml
from terra.env_generation.procedural_data import (
    generate_trenches_v2,
    generate_foundations_v2,
)
import terra.env_generation.postprocessing as postprocessing
from terra.env_generation.procedural_squares import generate_squares
import os
import yaml


def create_procedural_trenches(main_folder, config):
    # Load configurations from YAML
    resolution = config["resolution"]
    trenches_config = config["trenches"]
    difficulty_levels = trenches_config["difficulty_levels"]

    # Fix for loading tuples/lists correctly
    trenches_per_level = config["trenches"]["trenches_per_level"]
    corrected_trenches_per_level = [tuple(level) for level in trenches_per_level]

    n_imgs = config["n_imgs"]

    # Load new configurations for obstacles and non-dumpables
    n_obs_min = trenches_config["n_obs_min"]
    n_obs_max = trenches_config["n_obs_max"]
    size_obstacle_min = trenches_config["size_obstacle_min"]
    size_obstacle_max = trenches_config["size_obstacle_max"]
    n_nodump_min = trenches_config["n_nodump_min"]
    n_nodump_max = trenches_config["n_nodump_max"]
    size_nodump_min = trenches_config["size_nodump_min"]
    size_nodump_max = trenches_config["size_nodump_max"]

    for level, n_trenches in zip(difficulty_levels, corrected_trenches_per_level):
        save_folder = os.path.join(main_folder, "trenches", level)
        os.makedirs(save_folder, exist_ok=True)

        # Updated to use new configuration structure
        trench_dims_config = trenches_config["trench_dims"][level]
        trench_dims_min_ratio = trench_dims_config["min_ratio"]
        trench_dims_max_ratio = trench_dims_config["max_ratio"]

        trench_dims_min = (
            max(1, int(trench_dims_min_ratio[0] * trenches_config["img_edge_min"])),
            max(1, int(trench_dims_min_ratio[1] * trenches_config["img_edge_max"])),
        )
        trench_dims_max = (
            max(1, int(trench_dims_max_ratio[0] * trenches_config["img_edge_min"])),
            max(1, int(trench_dims_max_ratio[1] * trenches_config["img_edge_max"])),
        )

        generate_trenches_v2(
            n_imgs,
            trenches_config["img_edge_min"],
            trenches_config["img_edge_max"],
            trench_dims_min,
            trench_dims_max,
            n_trenches,  # Fixed to correctly pass the tuple/list
            resolution,
            save_folder,
            n_obs_min,
            n_obs_max,
            size_obstacle_min,
            size_obstacle_max,
            n_nodump_min,
            n_nodump_max,
            size_nodump_min,
            size_nodump_max,
        )


def create_foundations(config):
    """
    Creates procedural trenches across different difficulty levels using configurations from a YAML file.

    Parameters:
    - main_folder (str): The main directory where the trenches will be saved.
    - config_path (str): Path to the YAML configuration file.
    """
    foundation_config = config["foundations"]
    size = foundation_config["size"]
    dataset_path = foundation_config["dataset_rel_path"]

    generate_foundations_v2(
        size, package_dir + "/" + dataset_path, package_dir + "/data/terra/foundations"
    )
    print("Foundations created successfully.")


if __name__ == "__main__":
    config_path = "config/config.yaml"
    package_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    with open(package_dir + "/" + config_path, "r") as file:
        config = yaml.safe_load(file)
    # creeate config['dataset_rel_path'] folder
    os.makedirs("data/", exist_ok=True)
    os.makedirs("data/terra/", exist_ok=True)
    main_folder = "data/terra"
    create_procedural_trenches(main_folder, config)
    create_foundations(config)
    # now transform the pngs into npy arrays
    # sizes = [(16, 16), (32, 32), (64, 64)]#, (40, 80), (80, 160), (160, 320), (320, 640)]
    sizes = [(size, size) for size in config["sizes"]]
    npy_dataset_folder = package_dir + "/data/terra"
    print("npy_dataset_folder: ", npy_dataset_folder)
    for size in sizes:
        postprocessing.generate_dataset_terra_format(npy_dataset_folder, size)

        # Access the 'squares' configuration specifically
    squares_config = config["squares"]
    n_imgs = config[
        "n_imgs"
    ]  # 'n_imgs' is a top-level key, not nested within each square configuration
    base_save_folder = os.path.join(package_dir, "data/terra/train/squares")

    # Iterate over all configurations within 'squares'
    for config_name, config_values in squares_config.items():
        # Create a unique folder for each configuration
        save_folder = os.path.join(base_save_folder, config_name)
        os.makedirs(save_folder, exist_ok=True)

        # Generate squares for each configuration and save them in their respective folder
        generate_squares(n_imgs, config_values, save_folder)
        print("generated squares for ", config_name)
