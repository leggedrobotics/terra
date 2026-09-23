"""Import the 60d01307 V6/V7/V8 builders from ``tools/map_generation_60d01307``.

The V8 builders import ``tools.map_generation.*`` (package style) and their
generator siblings as top-level modules. Both names are bound to the pinned
snapshot here, before anything else imports them, and every resolved module is
checked to live in the snapshot directory.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

TERRA_ROOT = Path(__file__).resolve().parents[2]
PINNED = TERRA_ROOT / "tools" / "map_generation_60d01307"

if str(PINNED) not in sys.path:
    sys.path.insert(0, str(PINNED))
if str(TERRA_ROOT) not in sys.path:
    sys.path.insert(1, str(TERRA_ROOT))

if "tools.map_generation" not in sys.modules:
    importlib.import_module("tools")
    spec = importlib.util.spec_from_file_location(
        "tools.map_generation",
        PINNED / "__init__.py",
        submodule_search_locations=[str(PINNED)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules["tools.map_generation"] = package
    spec.loader.exec_module(package)

v8 = importlib.import_module("tools.map_generation.build_v8_combined_bank")
v7review = importlib.import_module("tools.map_generation.generate_v7_geometry_review")
controls = importlib.import_module("tools.map_generation.build_unconstrained_control_bank")
generator = importlib.import_module("generate_curriculum_bank")
prototypes = importlib.import_module("tools.map_generation.generate_prototypes")

for _name, _module in list(sys.modules.items()):
    if _module is None or not getattr(_module, "__file__", None):
        continue
    if _name.startswith("tools.map_generation") or _name in (
        "generate_curriculum_bank", "generate_prototypes_v9", "generate_prototypes_v8",
        "generate_prototypes_v7", "terra_service", "turn_dump", "terra_geom",
        "curriculum_taxonomy", "generate_prototypes",
    ):
        if Path(_module.__file__).resolve().parent != PINNED:
            raise RuntimeError(f"{_name} resolved outside the pinned snapshot: {_module.__file__}")
