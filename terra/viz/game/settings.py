MAP_TILES = (
    192  # 64 * 3, total number of tiles for a nice visualization (scales with MAP_EDGE)
)
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
    },
}
