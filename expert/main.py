# main.py
import csv
import numpy as np
from trajectory import load_reference_trajectory
from mh import run_parallel_chains
from visualizer import plot_trajectory_distribution, plot_control_points_3d, plot_samples_3d
from utils import load_kdtree_from_stl

if __name__ == "__main__":
    output_dir = "output/"

    # Load environment point cloud from STL file
    print("Loading environment point cloud")
    #stl_file = "data/office.stl"  # Path to your STL file
    pcd = o3d.io.read_point_cloud(file_path)
    point_cloud = np.asarray(pcd.points)
    kd_tree = cKDTree(point_cloud)

    # Load reference trajectory
    print("Loading reference trajectory...")
    times, ref_pos, ref_vel, ref_acc = load_reference_trajectory("data/reference_trajectory.csv")

    # Define state cost matrix Q (position and velocity weights)
    Q = np.zeros((6, 6))
    Q[:3, :3] = np.eye(3) * 1.0  # Position weights
    Q[3:, 3:] = np.eye(3) * 0.5  # Velocity weights

    # Prepare to save results for each 0.1 s interval
    all_samples = []

    # Open CSV for writing structured output
    print("Saving samples for each point...")
    with open(f"{output_dir}samples_by_time.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time", "sample", "cost"] + [f"cp{i}{coord}" for i in range(1, 4) for coord in ['x', 'y', 'z']])

        # Iterate over each 0.1s point in the reference trajectory
        for idx, (time, ref_pos_t, ref_vel_t, ref_acc_t) in enumerate(zip(times, ref_pos, ref_vel, ref_acc)):
            print(f"Running sampler for time {time:.1f} s (index {idx})...")

            # Define a sub-trajectory for the next 1.0s
            future_idx = idx + int(1.0 / 0.1)  # 1.0s forward in 0.1s steps
            ref_pos_future = ref_pos[idx:future_idx] if future_idx < len(ref_pos) else ref_pos[idx:]
            ref_vel_future = ref_vel[idx:future_idx] if future_idx < len(ref_vel) else ref_vel[idx:]

            # Run MH sampling for this point
            samples = run_parallel_chains(ref_pos_future, ref_vel_future, kd_tree, lambda_c=1000, Q=Q, n_samples=1000, n_chains=2)
            all_samples.extend(samples)

            # Write samples for this time step to the CSV
            for sample_idx, (cost, control_points) in enumerate(samples):
                writer.writerow([time, sample_idx + 1, cost] + control_points.flatten().tolist())

    # Visualize trajectory cost distribution for all samples
    print("Visualizing trajectory cost distribution...")
    plot_trajectory_distribution(all_samples, filename=f"{output_dir}trajectories_distribution.png")

    # Visualize control points in 3D
    print("Visualizing control points in 3D...")
    plot_control_points_3d(all_samples, filename=f"{output_dir}control_points_3d.png")

    print("All tasks completed successfully!")