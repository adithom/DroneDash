# collision.py
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

def quick_collision_check(positions, kd_tree, rq=0.2):
    # Check a few key points for collision
    # If any is too close to obstacle, return True immediately
    for pos in positions[::2]:  # check every other point for speed
        dist, idx = kd_tree.query(pos)
        if dist < rq:  # if too close, it's essentially a collision
            return True
    return False

def collision_cost(positions, kd_tree, rq=0.2):
    total_cost = 0.0
    dt = 0.1
    for pos in positions:
        dist, idx = kd_tree.query(pos)
        dc = dist
        if dc < 2*rq:
            cost_pt = -(dc**2)/(rq**2) + 4
        else:
            cost_pt = 0.0
        total_cost += cost_pt
    total_cost *= dt
    return total_cost
