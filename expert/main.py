# main.py
import os
import csv
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from trajectory import load_reference_trajectory
from mh import run_parallel_chains

if __name__ == "__main__":
    output_dir = "output/"
    os.makedirs(output_dir, exist_ok=True)

    # Load environment point cloud
    print("Loading environment point cloud")
    ply_file = "data/office_pointcloud.ply"  # Path to your PLY file
    pcd = o3d.io.read_point_cloud(ply_file)
    point_cloud = np.asarray(pcd.points)
    kd_tree = cKDTree(point_cloud)

    # Load reference trajectory
    print("Loading reference trajectory...")
    times, ref_pos, ref_vel = load_reference_trajectory("data/office_traj.csv")

    # Define state cost matrix Q (position and velocity weights)
    Q = np.zeros((6, 6))
    Q[:3, :3] = np.eye(3) * 1.0  # Position weights
    Q[3:, 3:] = np.eye(3) * 0.5  # Velocity weights

    # Prepare to save results for each 0.1 s interval
    all_samples = []

    # Open CSV for writing structured output
    # Updated header for four control points: cp1x, cp1y, cp1z, cp2x, cp2y, cp2z, cp3x, cp3y, cp3z, cp4x, cp4y, cp4z
    print("Saving samples for each point...")
    with open(f"{output_dir}samples_by_time.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time", "sample", "cost"] + [f"cp{i}{coord}" for i in range(1, 5) for coord in ['x', 'y', 'z']])

        # Iterate over each 0.1s point in the reference trajectory
        for idx, (time, ref_pos_t, ref_vel_t) in enumerate(zip(times, ref_pos, ref_vel)):
            print(f"Running sampler for time {time:.1f} s (index {idx})...")

            # Define a sub-trajectory for the next 1.0s
            future_idx = idx + int(1.0 / 0.1)  # 1.0 second forward in 0.1 s steps
            ref_pos_future = ref_pos[idx:future_idx] if future_idx < len(ref_pos) else ref_pos[idx:]
            ref_vel_future = ref_vel[idx:future_idx] if future_idx < len(ref_vel) else ref_vel[idx:]

            # Run MH sampling for this point
            samples = run_parallel_chains(ref_pos_future, ref_vel_future, kd_tree, lambda_c=1000, Q=Q, n_samples=1000, n_chains=2)
            all_samples.extend(samples)

            # Sort samples by cost (lowest first)
            samples.sort(key=lambda x: x[0])
            
            num_samples = len(samples)
            print(f"Generated {num_samples} samples for time {time:.1f}s. Selecting top 3.")
            
            # Select top three samples
            top_samples = samples[:3]

            # Write top three samples for this time step to the CSV
            for sample_idx, (cost, control_points) in enumerate(top_samples):
                writer.writerow([time, sample_idx + 1, cost] + control_points.flatten().tolist())

    print("All tasks completed successfully!")
