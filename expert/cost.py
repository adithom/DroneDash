# cost.py
import numpy as np
from collision import collision_cost

def compute_trajectory_cost(candidate_positions, candidate_velocities,
                            reference_positions, reference_velocities,
                            kd_tree, lambda_c, Q):
    """
    Compute the trajectory cost as a combination of deviation and collision costs.

    Parameters:
        candidate_positions: Nx3 array of sampled trajectory positions.
        candidate_velocities: Nx3 array of sampled trajectory velocities.
        reference_positions: Nx3 array of reference trajectory positions.
        reference_velocities: Nx3 array of reference trajectory velocities.
        kd_tree: KD-tree for collision checking.
        lambda_c: Weight for collision cost.
        Q: State cost matrix (size 6x6: for position and velocity weights).

    Returns:
        total_cost: Combined cost for the trajectory.
    """
    N = min(len(candidate_positions), len(reference_positions))
    dt = 0.1  # Time step

    # Position deviation cost
    pos_diff = candidate_positions[:N] - reference_positions[:N]
    pos_deviation_cost = 0.0
    for i in range(N):
        pos_deviation_cost += pos_diff[i].dot(Q[:3, :3]).dot(pos_diff[i])
    pos_deviation_cost *= dt

    # Velocity deviation cost
    vel_diff = candidate_velocities[:N] - reference_velocities[:N]
    vel_deviation_cost = 0.0
    for i in range(N):
        vel_deviation_cost += vel_diff[i].dot(Q[3:6, 3:6]).dot(vel_diff[i])
    vel_deviation_cost *= dt

    # Collision cost
    col_cost = collision_cost(candidate_positions[:N], kd_tree)

    # Total cost
    total_cost = lambda_c * col_cost + pos_deviation_cost + vel_deviation_cost
    return total_cost
