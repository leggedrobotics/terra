from digbench.procedural_squares_final import generate_squares, load_config
import os
from pathlib import Path

if __name__ == '__main__':
    package_path = Path(__file__).parent.parent
    config_path = os.path.join(package_path, 'digbench/config', 'squares_config.yaml')
    config = load_config(config_path)

    base_save_folder = os.path.join(package_path, 'data/generated_squares')

    # Iterate over all configurations
    for config_name, config_values in config.items():
        n_imgs = config_values['n_imgs']
        x_dim = config_values['x_dim']
        y_dim = config_values['y_dim']
        side_lens = config_values['side_lens']
        margin = config_values['margin']

        # Create a unique folder for each configuration
        save_folder = os.path.join(base_save_folder, config_name)
        os.makedirs(save_folder, exist_ok=True)

        # Generate squares for each configuration and save them in their respective folder
        generate_squares(n_imgs, x_dim, y_dim, side_lens, save_folder)