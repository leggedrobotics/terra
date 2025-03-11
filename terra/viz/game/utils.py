import pygame as pg


def rotate_triangle(center, points, scale, angle):
    vCenter = pg.math.Vector2(center)

    rotated_point = [pg.math.Vector2(p).rotate(angle) for p in points]

    triangle_points = [(vCenter + p * scale) for p in rotated_point]
    return triangle_points


def validate_angle_divisions(angle_count):
    if 360 % angle_count != 0:
        raise ValueError(f"Angle count must divide 360 evenly, got {angle_count}")
    if angle_count % 4 != 0:
        raise ValueError(f"Angle count must be a multiple of 4, got {angle_count}")
    return angle_count


def agent_base_to_angle(agent_base, base_angles=8):
    """
    Args:
        agent_base: The index of the base direction
        base_angles: Total number of possible base directions (default: 8)
    
    Returns:
        The angle in degrees
    """

    angle_increment = 360 / base_angles
    return (360 - (agent_base * angle_increment)) % 360


def agent_cabin_to_angle(agent_cabin, cabin_angles=8):
    """
    Args:
        agent_cabin: The index of the cabin direction
        cabin_angles: Total number of possible cabin directions (default: 8)
    
    Returns:
        The angle in degrees
    """
    angle_increment = 360 / cabin_angles
    return (360 - (agent_cabin * angle_increment)) % 360
