import pygame as pg


def rotate_triangle(center, points, scale, angle):
    vCenter = pg.math.Vector2(center)

    rotated_point = [pg.math.Vector2(p).rotate(angle) for p in points]

    triangle_points = [(vCenter + p * scale) for p in rotated_point]
    return triangle_points


def agent_base_to_angle(agent_base):
    if agent_base == 0:
        angle = 0
    elif agent_base == 1:
        angle = 270
    elif agent_base == 2:
        angle = 180
    elif agent_base == 3:
        angle = 90
    return angle


def agent_cabin_to_angle(agent_cabin):
    if agent_cabin == 0:
        angle = 0
    elif agent_cabin == 1:
        angle = 315
    elif agent_cabin == 2:
        angle = 270
    elif agent_cabin == 3:
        angle = 225
    elif agent_cabin == 4:
        angle = 180
    elif agent_cabin == 5:
        angle = 135
    elif agent_cabin == 6:
        angle = 90
    elif agent_cabin == 7:
        angle = 45
    return angle
