import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def visualize_and_save_map(data, folder, filename, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap="viridis", interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.savefig(folder / f"{filename}.jpg", bbox_inches="tight")
    plt.close()


def visualize_maps_recursive(
    base_folder, map_categories=["images", "occupancy", "dumpability"]
):
    base_folder = Path(base_folder)
    if not base_folder.exists():
        print(f"No folder found for {base_folder}, skipping.")
        return

    for category in map_categories:
        category_folder = base_folder / category
        if category_folder.exists():
            image_output_folder = category_folder / "visualized"
            image_output_folder.mkdir(parents=True, exist_ok=True)

            npy_files = list(category_folder.glob("*.npy"))
            for npy_file in tqdm(
                npy_files, desc=f"Processing {category} in {base_folder.name}"
            ):
                map_data = np.load(npy_file)
                print(map_data.shape)
                filename = npy_file.stem  # Removes the file extension
                visualize_and_save_map(
                    map_data,
                    image_output_folder,
                    filename,
                    f"{category.capitalize()}: {filename}",
                )

            print(
                f"Visualization complete for {category}. Images saved in {image_output_folder}"
            )
        else:
            # If the current category folder doesn't exist, check for subdirectories to recurse into
            for subfolder in base_folder.iterdir():
                if subfolder.is_dir():
                    visualize_maps_recursive(subfolder, map_categories)


if __name__ == "__main__":
    digbench_path = Path(__file__).resolve().parents[1]
    visualize_maps_recursive(
        "/home/lorenzo/git/terra_jax/terra/data/terra/train/foundations",
        map_categories=["occupancy"],
    )
