import numpy as np
from terra.viz.a_star import a_star  # Import the A* function

def test_a_star():
    """
    Test the A* algorithm with a simple grid and highlight the path.
    """
    # Create a simple 5x5 grid (0 = free, 1 = obstacle)
    grid = np.array([
        [0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0]
    ])

    # Define start and target positions
    start = (0, 0)  # Top-left corner
    target = (4, 4)  # Bottom-right corner

    # Run the A* algorithm
    path = a_star(grid, start, target)

    # Highlight the path in the grid
    highlighted_grid = grid.copy()
    if path:
        for x, y in path:
            highlighted_grid[x, y] = 9  # Mark the path with 9

    # Print the results
    print("Original Grid:")
    print(grid)
    print("\nHighlighted Grid (Path marked with 9):")
    print(highlighted_grid)
    print(f"\nStart: {start}")
    print(f"Target: {target}")
    print(f"Computed Path: {path}")

    # Verify the path is correct
    assert path is not None, "Path should not be None"
    assert path[0] == start, "Path should start at the start position"
    assert path[-1] == target, "Path should end at the target position"

    # Check that all intermediate steps are valid
    for i in range(1, len(path)):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        assert abs(x1 - x2) + abs(y1 - y2) == 1, "Path steps should be adjacent"
        assert grid[x2, y2] == 0, "Path should not pass through obstacles"

    print("\nTest passed!")


if __name__ == "__main__":
    test_a_star()