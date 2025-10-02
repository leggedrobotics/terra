MAP_TILES = (
    192  # 64 * 3, total number of tiles for a nice visualization (scales with MAP_EDGE)
)

def darken_color(color, factor=0.7):
    """
    Make a color darker by multiplying RGB values by a factor.
    Args:
        color: RGB tuple like (255, 140, 0)
        factor: Darkness factor (0.0 = black, 1.0 = original color)
    Returns:
        Darker RGB tuple
    """
    if isinstance(color, tuple) and len(color) == 3:
        return tuple(int(c * factor) for c in color)
    return color

COLORS = {
    0: "#cfcfcf",  # neutral
    5: "#ccffcc",  # final dumping area to terminate the episode (very light green)
    3: "#ab9f95",  # non-dumpable (e.g. road)
    4: "#8800ff",  # to dig
    2: "#000000",  # obstacle
    1: "#002B5B",  # action map dump
    -1: "#26bd6c",  # action map dug
    6: "#ff6b6b",  # interaction mask (dig/dump cones) - bright red
    "agent_body": (0, 43, 91),  # Blue for tracked agents
    "agent_cabin": {
        "loaded": (165, 115, 75),
        "not_loaded": (234, 84, 85),
    },
    "skid_steer_body": (255, 140, 0),  # Orange for skid steer body
    "skid_steer_cabin": {
        "loaded": (255, 215, 0),  # Gold when loaded
        "not_loaded": (255, 165, 0),  # Orange when not loaded
    },
    "shovel": {
        "lowered": (139, 69, 19),      # Brown - shovel on ground 
        "lifted": (192, 192, 192),     # Silver - shovel lifted
    }
}
