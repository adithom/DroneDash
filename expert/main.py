# main.py
import csv
import numpy as np
from scipy.spatial import cKDTree
from reference_loader import load_reference_trajectory
from mh_sampler import run_parallel_chains
from visualize import plot_trajectory_distribution, plot_samples_3d

if __name__ == "__main__":
    # Load environment pointcloud and build KD-tree
    # Here just a dummy environment:
    environment_points = np.random.rand(5000,3)*10.0
    kd_tree = cKDTree(environment_points)

    # Load reference trajectory
    times, ref_pos, ref_vel, ref_acc = load_reference_trajectory("environment/reference_trajectory.csv")

    # Run M-H sampling with parallelization
    samples = run_parallel_chains(ref_pos, kd_tree, n_samples=10000, n_chains=4)

    # Extract top 3
    best_3 = samples[:3]

    # Save top 3 trajectories
    with open("output/top_trajectories.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["cost"] + [f"cp{i}{coord}" for i in range(1,4) for coord in ['x','y','z']])
        for c, cps in best_3:
            writer.writerow([c]+cps.flatten().tolist())

    # Visualize distribution
    plot_trajectory_distribution(samples)
    plot_samples_3d(samples)
