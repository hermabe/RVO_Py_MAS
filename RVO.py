import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class VelocityObstacle:
    apex: np.array
    bound_left: np.array
    bound_right: np.array
    distance_to: float
    radius: float


def velocity_towards_goal(position: np.array, goal: np.array, max_velocity: float) -> np.array:
    GOAL_MARGIN = 0.1
    direction = goal - position
    norm = np.linalg.norm(direction)
    if norm < GOAL_MARGIN:
        return np.zeros(2)
    return direction * max_velocity/norm


def create_velocity_obstacles(position: np.array, obstacles: list) -> List[VelocityObstacle]:
    velocity_obstacles = []
    for obstacle in obstacles:
        radius = max(obstacle.width, obstacle.length)
        obstacle_velocity = obstacle.direction * obstacle.speed
        vo_apex = position + obstacle_velocity
        obstacle_vector = obstacle.position - position
        angle_to_obstacle = np.arctan2(obstacle_vector[1], obstacle_vector[0])

        distance_to_obstacle = np.linalg.norm(obstacle_vector)
        distance_to_obstacle = max(distance_to_obstacle, 2*radius)

        theta_BAort = np.arcsin(2*radius/distance_to_obstacle)
        theta_ort_left = angle_to_obstacle+theta_BAort
        bound_left = np.array([np.cos(theta_ort_left), np.sin(theta_ort_left)])
        theta_ort_right = angle_to_obstacle-theta_BAort
        bound_right = np.array([np.cos(theta_ort_right), np.sin(theta_ort_right)])

        velocity_obstacles.append(VelocityObstacle(vo_apex, bound_left,
                                                   bound_right, distance_to_obstacle, 2*radius))
    return velocity_obstacles


def theta_dif_right_left(velocity_obstacle: VelocityObstacle, new_velocity: np.array, position: np.array, return_dif: bool = False) -> Tuple[float, float, float]:
    diff = new_velocity + position - velocity_obstacle.apex
    theta_dif = np.arctan2(diff[1], diff[0])
    theta_right = np.arctan2(velocity_obstacle.bound_right[1], velocity_obstacle.bound_right[0])
    theta_left = np.arctan2(velocity_obstacle.bound_left[1], velocity_obstacle.bound_left[0])
    if return_dif:
        return theta_dif, theta_right, theta_left, diff
    return theta_dif, theta_right, theta_left


def check_is_suitable(velocity_obstacles: List[VelocityObstacle], new_velocity: np.array, position: np.array) -> bool:
    # If in any velocity obstacle, it is unsuitable
    for velocity_obstacle in velocity_obstacles:
        theta_dif, theta_right, theta_left = theta_dif_right_left(velocity_obstacle, new_velocity, position)
        if in_between(theta_right, theta_dif, theta_left):
            return False
    return True


def find_suitable_unsuitable_velocities(velocity_obstacles: List[VelocityObstacle], velocity: np.array, position: np.array) -> Tuple[List[np.array], List[np.array]]:
    velocity_norm = np.linalg.norm(velocity)
    suitable_velocities = []
    unsuitable_velocities = []
    for theta in np.arange(0, 2*math.pi, 0.1):
        for rad in np.arange(0.02, velocity_norm+0.02, velocity_norm/5.0):
            new_velocity = rad*np.array([np.cos(theta), np.sin(theta)])
            if check_is_suitable(velocity_obstacles, new_velocity, position):
                suitable_velocities.append(new_velocity)
            else:
                unsuitable_velocities.append(new_velocity)

    return suitable_velocities, unsuitable_velocities


def intersect(position: np.array, desired_velocity: np.array, velocity_obstacles: List[VelocityObstacle]) -> np.array:
    suitable_velocities, unsuitable_velocities = find_suitable_unsuitable_velocities(
        velocity_obstacles, desired_velocity, position)

    # Add desired velocity as well
    if check_is_suitable(velocity_obstacles, desired_velocity, position):
        suitable_velocities.append(desired_velocity)
    else:
        unsuitable_velocities.append(desired_velocity)

    if suitable_velocities:
        # Return the suitable velocity closest to the desired velocity
        return min(suitable_velocities, key=lambda v: np.linalg.norm(v - desired_velocity))
    else:
        return intersect_unsuitable(unsuitable_velocities, velocity_obstacles, position, desired_velocity)


def intersect_unsuitable(unsuitable_velocities: List[np.array], velocity_obstacles: List[VelocityObstacle], position: np.array, desired_velocity: np.array) -> np.array:
    WT = 0.2
    best_velocity = unsuitable_velocities[0]
    best_value = float('inf')

    for unsuitable_velocity in unsuitable_velocities:
        tc = []
        for velocity_obstacle in velocity_obstacles:
            rad = velocity_obstacle.radius
            theta_dif, theta_right, theta_left, dif = theta_dif_right_left(
                velocity_obstacle, unsuitable_velocity, position, return_dif=True)
            if in_between(theta_right, theta_dif, theta_left):
                small_theta = abs(theta_dif-0.5*(theta_left+theta_right))  # Some angle to the left of the VO?
                what = abs(velocity_obstacle.distance_to*np.sin(small_theta))
                rad = max(rad, what)
                big_theta = np.arcsin(what/rad)
                dist_tg = abs(velocity_obstacle.distance_to*np.cos(small_theta))-abs(rad*np.cos(big_theta))
                dist_tg = max(0, dist_tg)
                tc_v = dist_tg/np.linalg.norm(dif)
                tc.append(tc_v)

        value = WT/(min(tc) + 0.001) + np.linalg.norm(unsuitable_velocity - desired_velocity)
        if value < best_value:
            best_velocity = unsuitable_velocity
            best_value = value
    return best_velocity


def VO_update(position: np.array, desired_velocity: np.array, obstacles: list) -> np.array:
    velocity_obstacles = create_velocity_obstacles(position, obstacles)
    return intersect(position, desired_velocity, velocity_obstacles)


def in_between(theta_right, theta_dif, theta_left):
    if abs(theta_right - theta_left) <= math.pi:
        if theta_right <= theta_dif <= theta_left:
            return True
        else:
            return False
    else:
        if (theta_left < 0) and (theta_right > 0):
            theta_left += 2*math.pi
            if theta_dif < 0:
                theta_dif += 2*math.pi
            if theta_right <= theta_dif <= theta_left:
                return True
            else:
                return False
        if (theta_left > 0) and (theta_right < 0):
            theta_right += 2*math.pi
            if theta_dif < 0:
                theta_dif += 2*math.pi
            if theta_left <= theta_dif <= theta_right:
                return True
            else:
                return False
