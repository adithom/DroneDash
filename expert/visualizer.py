# visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import os

def ensure_directory(filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_cost_distribution_by_time(samples, filename="output/cost_distribution_by_time.png"):
    """
    Plot a histogram of trajectory costs, grouped by time step.
    """
    ensure_directory(filename)
    times = [s[0] for s in samples]  # Extract times
    costs = [s[1] for s in samples]  # Extract costs

    plt.figure()
    scatter = plt.scatter(times, costs, c=times, cmap="viridis", alpha=0.6, s=10)
    plt.colorbar(scatter, label="Time (s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Cost")
    plt.title("Trajectory Costs by Time Step")
    plt.savefig(filename)
    plt.close()

def plot_control_points_3d_by_time(samples, filename="output/control_points_3d_by_time.png"):
    """
    Plot 3D control points, grouped by time step.
    """
    ensure_directory(filename)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for sample in samples:
        time, _, _, cp1x, cp1y, cp1z, cp2x, cp2y, cp2z, cp3x, cp3y, cp3z = sample
        control_points = np.array([[cp1x, cp1y, cp1z], [cp2x, cp2y, cp2z], [cp3x, cp3y, cp3z]])
        ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2], alpha=0.6, label=f"t={time:.1f}")

    ax.set_title("Control Points by Time Step")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(filename)
    plt.close()

def plot_trajectories(samples, ref_positions, filename="output/trajectories_by_time.png"):
    """
    Plot trajectories defined by the samples, overlaid with the reference trajectory.
    """
    ensure_directory(filename)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot reference trajectory
    ref_positions = np.array(ref_positions)
    ax.plot(ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2], color="blue", linewidth=2, label="Reference")

    # Plot sampled trajectories
    for sample in samples:
        time, _, _, cp1x, cp1y, cp1z, cp2x, cp2y, cp2z, cp3x, cp3y, cp3z = sample
        control_points = np.array([[cp1x, cp1y, cp1z], [cp2x, cp2y, cp2z], [cp3x, cp3y, cp3z]])
        ax.plot(control_points[:, 0], control_points[:, 1], control_points[:, 2], alpha=0.6)

    ax.set_title("Sampled Trajectories by Time Step")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.savefig(filename)
    plt.close()
