import numpy as np
import math
import pygame as pg
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

    def create_shovel(self, center_x, center_y, base_angle_degrees):
        """
        Create a shovel (rectangle) for skid steer agents.
        The shovel is positioned in the same direction as the cabin triangle.
        """
        # Shovel dimensions - wider than the agent base
        shovel_depth = (self.height * self.tile_size) * 0.4    # Increased from 0.25 to 0.4 (40% of agent height)
        shovel_width = (self.width * self.tile_size) * 1.05     # 120% of agent width (wide perpendicular to movement)
        
        # Calculate shovel position using the same method as cabin triangle
        agent_half_height = (self.height * self.tile_size) / 2
        shovel_offset_distance = agent_half_height + shovel_depth / 2 + 1  # Small gap
        
        # Create shovel center point in the same direction as cabin front point
        # Cabin front point is at (3/scaling, 0) in local coordinates
        scaling = self.tile_size / 3
        cabin_front_direction = pg.math.Vector2(3 / scaling, 0)
        
        # Normalize and scale to shovel offset distance
        cabin_front_normalized = cabin_front_direction.normalize()
        shovel_center_local = cabin_front_normalized * shovel_offset_distance
        
        # Rotate the shovel center using the same method as cabin
        shovel_center_rotated = shovel_center_local.rotate(base_angle_degrees)
        
        # Calculate final shovel center position
        center_vector = pg.math.Vector2(center_y, center_x)  # Note: (y, x) order
        shovel_center_world = center_vector + shovel_center_rotated
        
        # Create shovel rectangle corners relative to shovel center
        half_depth = shovel_depth / 2   # Half depth in movement direction 
        half_width = shovel_width / 2   # Half width perpendicular to movement
        
        shovel_local_points = [
            pg.math.Vector2(-half_depth, -half_width),  # back-left
            pg.math.Vector2(half_depth, -half_width),   # front-left  
            pg.math.Vector2(half_depth, half_width),    # front-right
            pg.math.Vector2(-half_depth, half_width)    # back-right
        ]
        
        # Rotate and position all shovel points
        shovel_vertices = []
        for local_point in shovel_local_points:
            # Rotate the local point
            rotated_point = local_point.rotate(base_angle_degrees)
            # Position relative to shovel center
            final_point = shovel_center_world + rotated_point
            shovel_vertices.append((final_point.x, final_point.y))
        
        return shovel_vertices

    def create_agent(self, px_center, py_center, angle_base, angle_cabin, loaded, agent_type=0, shovel_lifted=0):
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
        
        # Helper to create an oriented rectangle given a local center offset (forward is +X in cabin space here)
        def oriented_rect(center_xy, base_deg, rect_width_px, rect_depth_px, forward_offset_px):
            # Local center relative to agent center in forward direction
            local_cx = forward_offset_px
            local_cy = 0.0
            # Rotate local center by base angle
            ang = math.radians(base_deg)
            off_x = local_cx * math.cos(ang) - local_cy * math.sin(ang)
            off_y = local_cx * math.sin(ang) + local_cy * math.cos(ang)
            cx = center_xy[0] + off_x
            cy = center_xy[1] + off_y

            half_w = rect_width_px / 2.0
            half_d = rect_depth_px / 2.0
            # Rectangle in local coordinates (depth along forward axis)
            local_pts = [
                (-half_d, -half_w),  # back-left
                ( half_d, -half_w),  # front-left
                ( half_d,  half_w),  # front-right
                (-half_d,  half_w),  # back-right
            ]
            verts = []
            for lx, ly in local_pts:
                rx = lx * math.cos(ang) - ly * math.sin(ang)
                ry = lx * math.sin(ang) + ly * math.cos(ang)
                verts.append((center_xy[0] + rx + off_x, center_xy[1] + ry + off_y))
            return verts

        # Create cabin triangle by default (non-truck)
        scaling = self.tile_size / 3
        points = [
            (3 / scaling, 0),
            (-1.5 / scaling, -1.5 / scaling),
            (-1.5 / scaling, 1.5 / scaling),
        ]
        agent_cabin = rotate_triangle(
            (center_y, center_x), points, self.tile_size, global_cabin_angle
        )

        # Choose colors based on agent type
        if agent_type == 2:  # Skid steer
            body_color = COLORS["skid_steer_body"]
            cabin_color = COLORS["skid_steer_cabin"]["loaded"] if loaded else COLORS["skid_steer_cabin"]["not_loaded"]
        elif agent_type == 3:  # Truck
            # Dark green shades
            body_color = (0, 90, 40)
            cabin_color = (0, 110, 50)
        else:  # Tracked or wheeled (default)
            body_color = COLORS["agent_body"]
            cabin_color = COLORS["agent_cabin"]["loaded"] if loaded else COLORS["agent_cabin"]["not_loaded"]

        # Truck rendering override: base same size as other agents + simple rear cabin
        if agent_type == 3:
            total_depth = self.height * self.tile_size
            rect_width = self.width * self.tile_size
            # Use the standard body rectangle as the truck base (same dims as other agents)
            bed_vertices = agent_body
            # Add a cabin at the back end of the base, flush with the end
            # Cabin: same length (shorter side) as agent, 25% of agent width (longer side)
            cabin_length = self.height * self.tile_size  # same as agent length (shorter side)
            cabin_width = rect_width * 0.25  # 25% of agent width (longer side)
            # Position cabin flush at the back end: offset = -(half_base_length + half_cabin_length)
            cabin_offset = -(total_depth / 2.0 + cabin_length / 2.0)
            truck_cabin_vertices = oriented_rect((center_y, center_x), base_angle_degrees, cabin_length, cabin_width, cabin_offset)

            out = {
                "body": {
                    "vertices": bed_vertices,
                    "width": w,
                    "height": h,
                    "color": body_color,
                },
                "cabin": {
                    "vertices": truck_cabin_vertices,
                    "color": cabin_color,
                },
            }
        else:
            out = {
                "body": {
                    "vertices": agent_body,
                    "width": w,
                    "height": h,
                    "color": body_color,
                },
                "cabin": {
                    "vertices": agent_cabin,
                    "color": cabin_color,
                },
            }
        
        # Add shovel for skid steer agents
        if agent_type == 2:
            shovel_vertices = self.create_shovel(center_x, center_y, base_angle_degrees)
            
            # Determine shovel color based on lifted state only
            if shovel_lifted == 0:
                shovel_color = COLORS["shovel"]["lowered"]  # Brown - shovel on ground
            else:
                shovel_color = COLORS["shovel"]["lifted"]   # Silver - shovel lifted
            
            out["shovel"] = {
                "vertices": shovel_vertices,
                "color": shovel_color,
            }
        
        return out

    def update(self, agent_pos, base_dir, cabin_dir, loaded, agent_type=0, shovel_lifted=0):
        agent_pos = np.asarray(agent_pos, dtype=np.int32)
        base_dir = np.asarray(base_dir, dtype=np.int32)
        cabin_dir = np.asarray(cabin_dir, dtype=np.int32)
        loaded = np.asarray(loaded, dtype=bool)
        agent_type = np.asarray(agent_type, dtype=np.int32)
        shovel_lifted = np.asarray(shovel_lifted, dtype=np.int32)
        self.agent = self.create_agent(
            agent_pos[0].item(),
            agent_pos[1].item(),
            base_dir.item(),
            cabin_dir.item(),
            loaded.item(),
            agent_type.item(),  # Pass agent type
            shovel_lifted.item(),  # Pass shovel state
        )
