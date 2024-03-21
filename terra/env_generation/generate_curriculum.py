from terra.env_generation.procedural_squares import generate_squares, load_config
import os
from pathlib import Path

if __name__ == '__main__':
    package_path = Path(__file__).parent.parent.parent
    config_path = os.path.join(package_path, 'config', 'config.yaml')
    config = load_config(config_path)
    print("config: ", config)

    base_save_folder = os.path.join(package_path, 'data/generated_squares')

    # Access the 'squares' configuration specifically
    squares_config = config['squares']
    n_imgs = config['n_imgs']  # 'n_imgs' is a top-level key, not nested within each square configuration

    # Iterate over all configurations within 'squares'
    for config_name, config_values in squares_config.items():
        x_dim = config_values['x_dim']
        y_dim = config_values['y_dim']
        side_lens = config_values['side_lens']
        margin = config_values['margin']

        # Create a unique folder for each configuration
        save_folder = os.path.join(base_save_folder, config_name)
        os.makedirs(save_folder, exist_ok=True)

        # Generate squares for each configuration and save them in their respective folder
        generate_squares(n_imgs, x_dim, y_dim, side_lens, save_folder)