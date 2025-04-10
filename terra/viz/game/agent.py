import numpy as np
import math
from .settings import COLORS
from .utils import agent_base_to_angle
from .utils import agent_cabin_to_angle
from .utils import rotate_triangle


class Agent:
    def __init__(self, width, height, tile_size, angles_base, angles_cabin) -> None:
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.angles_base = angles_base
        self.angles_cabin = angles_cabin

    def create_agent(self, px_center, py_center, angle_base, angle_cabin, loaded):
    # Convert angle_base index to degrees using the util function
        base_angle_degrees = agent_base_to_angle(angle_base, self.angles_base)
        
        # Calculate center position in pixels
        center_x = px_center * self.tile_size
        center_y = py_center * self.tile_size
        
        # Calculate half-dimensions for the body
        half_width = (self.width * self.tile_size) / 2
        half_height = (self.height * self.tile_size) / 2
        
        # Create the four corners of the rectangle (unrotated)
        rect_points = [
            (-half_width, -half_height),  # top-left
            (half_width, -half_height),   # top-right
            (half_width, half_height),    # bottom-right
            (-half_width, half_height)    # bottom-left
        ]
        
        # Rotate and translate each point
        agent_body = []
        for x, y in rect_points:
            # Convert to radians for trigonometric calculations
            angle_rad = math.radians(base_angle_degrees)
            # Apply rotation
            rotated_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            rotated_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
            # Translate to actual position
            agent_body.append((center_y + rotated_x, center_x + rotated_y))
        
        # Use the actual dimensions for the agent
        w = self.width
        h = self.height

        # Calculate cabin angle (global orientation)
        cabin_relative_degrees = agent_cabin_to_angle(angle_cabin, self.angles_cabin)
        global_cabin_angle = (cabin_relative_degrees + base_angle_degrees) % 360
        
        # Create cabin triangle
        scaling = self.tile_size / 3
        points = [
            (3 / scaling, 0),
            (-1.5 / scaling, -1.5 / scaling),
            (-1.5 / scaling, 1.5 / scaling),
        ]
        agent_cabin = rotate_triangle(
            (center_y, center_x), points, self.tile_size, global_cabin_angle
        )
        # Compute the front marker position based on the base's direction
        # front_marker_offset = {
        #     0: (w * self.tile_size, h * self.tile_size // 2),  # Right
        #     1: (w * self.tile_size // 2, 0),  # Up
        #     2: (0, h * self.tile_size // 2),  # Left
        #     3: (w * self.tile_size // 2, h * self.tile_size),  # Down
        # }

        # px_front, py_front = front_marker_offset[angle_base]

        # # Add the front marker
        # front_marker = [
        #     (agent_body[0][0] + px_front, agent_body[0][1] + py_front)
        # ]
        
        out = {
            "body": {
                "vertices": agent_body,
                "width": w,
                "height": h,
                "color": COLORS["agent_body"],
            },
            "cabin": {
                "vertices": agent_cabin,
                "color": COLORS["agent_cabin"]["loaded"]
                if loaded
                else COLORS["agent_cabin"]["not_loaded"],
            },
            # "front_marker": {  
            #     "vertices": front_marker,
            #     "color": (255, 255, 0),  
            # },
            "angle_base": angle_base,
        }

        return out

    def update(self, agent_pos, base_dir, cabin_dir, loaded):
        agent_pos = np.asarray(agent_pos, dtype=np.int32)
        base_dir = np.asarray(base_dir, dtype=np.int32)
        cabin_dir = np.asarray(cabin_dir, dtype=np.int32)
        loaded = np.asarray(loaded, dtype=bool)
        self.agent = self.create_agent(
            agent_pos[0].item(),
            agent_pos[1].item(),
            base_dir.item(),
            cabin_dir.item(),
            loaded.item(),
        )
