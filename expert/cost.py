# cost_function.py
import numpy as np
from collision import collision_cost

def compute_trajectory_cost(candidate_positions, reference_positions, kd_tree, lambda_c, Q):
    # candidate_positions: Nx3
    # reference_positions: Nx3 (same length)
    N = min(len(candidate_positions), len(reference_positions))
    dt = 0.1

    diff = candidate_positions[:N] - reference_positions[:N]
    # Deviation cost = ∫ (τ(t)-τref(t))^T Q (τ(t)-τref(t)) dt approximated by sum
    deviation_cost = 0.0
    for i in range(N):
        deviation_cost += diff[i].dot(Q).dot(diff[i])
    deviation_cost = deviation_cost * dt

    # Collision cost from environment
    col_cost = collision_cost(candidate_positions[:N], kd_tree)

    total_cost = lambda_c * col_cost + deviation_cost
    return total_cost