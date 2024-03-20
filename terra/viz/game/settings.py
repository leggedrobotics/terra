TILE_SIZE = 3  # 10 for manual testing, 3 for multi-environment visualization
MAP_EDGE = 64  # the number of tiles in the map's edge
AGENT_DIMS = (9, 5)  # (width, height) in pixels

COLORS = {
    0: "#cfcfcf",  # neutral
    5: "#E4DCCF",  # final dumping area to terminate the episode
    3: "#ab9f95",  # non-dumpable (e.g. road)
    4: "#8800ff",  # to dig
    2: "#000000",  # obstacle
    1: "#002B5B",  # action map dump
    -1: "#26bd6c",  # action map dug
    "agent_body": (0, 43, 91),
    "agent_cabin": {
        "loaded": (234, 84, 85),
        "not_loaded": (234, 84, 85),
    }
}
