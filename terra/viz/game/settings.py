TILE_SIZE = 10  # 10 for manual testing, 3 for multi-environment visualization
MAP_EDGE = 64  # the number of tiles in the map's edge
AGENT_DIMS = (9, 5)  # (width, height) in pixels

COLORS = {
    0: "#E4DCCF",
    1: "#002B5B",
    -1: "#C98474",
    2: "#EA5455",  # obstacle
    3: "#A7D2CB",  # non-dumpable
    4: "#d1c9bc",  # to dig
    "agent_body": (0, 43, 91),
    "agent_cabin": {
        "loaded": (234, 84, 85),
        "not_loaded": (234, 84, 85),
    }
}
