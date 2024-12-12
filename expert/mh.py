# mh.py
import numpy as np
import random
from utils import generate_bspline_trajectory
from cost import compute_trajectory_cost
from collision import quick_collision_check

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < 1e-8:
        return r, 0.0, 0.0
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def propose_control_points(current_control_points, std_dev):
    proposed = current_control_points.copy()
    for cp_idx in range(1, len(proposed)):
        base_pt = proposed[cp_idx]
        r, theta, phi = cartesian_to_spherical(base_pt[0], base_pt[1], base_pt[2])
        theta_new = theta + np.random.uniform(-std_dev*np.pi/180, std_dev*np.pi/180)
        phi_new = phi + np.random.uniform(-std_dev*np.pi/180, std_dev*np.pi/180)
        x_new, y_new, z_new = spherical_to_cartesian(r, theta_new, phi_new)
        proposed[cp_idx] = [x_new, y_new, z_new]
    return proposed

def metropolis_hastings_chain(ref_positions, ref_velocities, kd_tree, lambda_c=1000, Q=np.eye(6), n_samples=50000, rand_theta_std=[2,5,10], initial_control_points=None, max_retries=5):
    # Initial guess
    if initial_control_points is None:
        N = len(ref_positions)
        p0 = ref_positions[0]
        p1 = ref_positions[N//3]      # 1/3 along the trajectory
        p2 = ref_positions[2*N//3]    # 2/3 along the trajectory
        p3 = ref_positions[-1]
        current_control_points = np.array([p0, p1, p2, p3])
    else:
        current_control_points = initial_control_points.copy()

    # Initial cost
    _, pos_samples, vel_samples, acc_samples = generate_bspline_trajectory(current_control_points)
    current_cost = compute_trajectory_cost(pos_samples, vel_samples, ref_positions, ref_velocities, kd_tree, lambda_c, Q)
    current_score = np.exp(-current_cost)

    accepted_samples = [(current_cost, current_control_points.copy())]

    n_third = n_samples // 3

    for i in range(n_samples):
        # Determine std_dev based on which third of samples we are in
        if i < n_third:
            std_dev = rand_theta_std[0]
        elif i < 2*n_third:
            std_dev = rand_theta_std[1]
        else:
            std_dev = rand_theta_std[2]

        # Try multiple times to get a non-colliding proposal
        for _ in range(max_retries):
            proposed_control_points = propose_control_points(current_control_points, std_dev)
            _, pos_samples_p, vel_samples_p, acc_samples_p = generate_bspline_trajectory(proposed_control_points)
            
            # Check collision
            if not quick_collision_check(pos_samples_p, kd_tree):
                # If no collision, evaluate cost and accept/reject
                candidate_cost = compute_trajectory_cost(pos_samples_p, vel_samples_p, ref_positions, ref_velocities, kd_tree, lambda_c, Q)
                candidate_score = np.exp(-candidate_cost)
                
                alpha = min(1.0, candidate_score/current_score)
                if random.random() < alpha:
                    current_control_points = proposed_control_points
                    current_score = candidate_score
                    accepted_samples.append((candidate_cost, current_control_points.copy()))
                # Break out of retries loop since we made a proposal attempt
                break
            # If there's a collision, try again (up to max_retries)

    accepted_samples.sort(key=lambda x: x[0])
    return accepted_samples

def run_parallel_chains(ref_positions, ref_velocities, kd_tree,
                        lambda_c=1000, Q=np.eye(6), 
                        n_samples=50000, n_chains=4):
    # Parallelize using multiprocessing or joblib
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_chains)(
        delayed(metropolis_hastings_chain)(ref_positions, ref_velocities, kd_tree, lambda_c, Q, n_samples)
        for _ in range(n_chains)
    )
    # Combine results
    combined = []
    for res in results:
        combined.extend(res)
    combined.sort(key=lambda x: x[0])
    return combined
