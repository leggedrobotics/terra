import heapq
import numpy as np

def inflate_obstacles(grid, buffer_size):
    """
    Inflates obstacles in the grid by marking cells within a buffer zone as non-traversable.

    Args:
        grid (np.ndarray): The original grid.
        buffer_size (int): The number of cells around each obstacle to mark as non-traversable.

    Returns:
        np.ndarray: The inflated grid.
    """
    inflated_grid = grid.copy()
    rows, cols = grid.shape

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != 0:  # If the cell is an obstacle
                for dr in range(-buffer_size, buffer_size + 1):
                    for dc in range(-buffer_size, buffer_size + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            inflated_grid = inflated_grid.at[nr, nc].set(1)  # Mark as non-traversable

    return inflated_grid

def a_star(grid, start, target, buffer_size=1):
    """
    A* algorithm to find the shortest path in a grid.

    Args:
        grid: 2D numpy array representing the map (0 = free, 1 = obstacle).
        start: Tuple (x, y) representing the start position.
        target: Tuple (x, y) representing the target position.

    Returns:
        List of tuples representing the path from start to target, or None if no path exists.
    """
    # Inflate obstacles
    grid = inflate_obstacles(grid, buffer_size)

    def heuristic(a, b):
        # Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    rows, cols = grid.shape

    # Ensure start and target are traversable
    if grid[start[0], start[1]] != 0 or grid[target[0], target[1]] != 0:
        print("Start or target position is not traversable.")
        return None

    # Handle edge case where start == target
    if start == target:
        return [start]

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, target)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target:
            # Reconstruct the path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Add diagonals if needed
        ]
        for neighbor in neighbors:
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 0:  # Check for traversable area
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, target)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    print("No path found.")
    return None  # No path found

def compute_path(state, start, target):
    """
    Compute the path from start to target using the A* algorithm.

    Args:
        state: The current game state object.
        start: Tuple (x, y) representing the start position.
        target: Tuple (x, y) representing the target position.

    Returns:
        A tuple containing:
        - path: List of tuples representing the computed path, or None if no path is found.
        - highlighted_grid: The grid with the path highlighted (marked with 9).
    """
    target_map = state.world.target_map.map[0]                      # Extract the target map
    #target_map = target_map.at[start[0], start[1]].set(7)           # Mark the start position
    #target_map = target_map.at[target[0], target[1]].set(8)         # Mark the target position
    traversability_mask = state.world.traversability_mask.map[0]    # Extract the traversability mask

    # Combine the maps
    combined_grid = target_map.copy()

    # Ensure target map values are correctly set for A* logic
    combined_grid = combined_grid.at[target_map == -1].set(0)  # Digging target (traversable)
    combined_grid = combined_grid.at[target_map == 1].set(0)   # Dumping target (traversable)
    combined_grid = combined_grid.at[target_map == 0].set(0)   # Free space (traversable)

    # Apply traversability mask
    combined_grid = combined_grid.at[traversability_mask == 1].set(1)  # Non-traversable
  
    # Mark the start and target positions
    #combined_grid = combined_grid.at[start[0], start[1]].set(0)    # Mark the start position back to 0
    #combined_grid = combined_grid.at[target[0], target[1]].set(0)  # Mark the target position back to 0

    # Run the A* algorithm
    path = a_star(combined_grid, start, target, buffer_size=2)
    
    return path, combined_grid

def simplify_path(path):
    """
    Simplify a path by reducing it to straight lines and minimizing turns.

    Args:
        path (list of tuples): The original path as a list of (x, y) positions.

    Returns:
        list of tuples: The simplified path.
    """
    if not path:
        return []

    # Initialize the simplified path with the first point
    simplified_path = [path[0]]

    # Iterate through the path and keep only points where the direction changes
    for i in range(1, len(path) - 1):
        prev_point = path[i - 1]
        curr_point = path[i]
        next_point = path[i + 1]

        # Calculate the direction vectors
        dir1 = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
        dir2 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])

        # If the direction changes, add the current point to the simplified path
        if dir1 != dir2:
            simplified_path.append(curr_point)

    # Add the last point to the simplified path
    simplified_path.append(path[-1])

    return simplified_path