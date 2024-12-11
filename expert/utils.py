# utils.py
import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh
from scipy.spatial import cKDTree
from sklearn.utils import shuffle

def load_kdtree_from_stl(file_path, downsample_rate=None):
    """
    Loads a KD-tree from an STL file by extracting the vertices of the mesh and optionally downsampling.
    
    Parameters:
        file_path (str): Path to the STL file.
        downsample_rate (float): Fraction of points to retain for downsampling (e.g., 0.1 for 10%).
    
    Returns:
        cKDTree: A KD-tree constructed from the processed vertices of the STL mesh.
    """
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(file_path)
    
    # Extract unique vertices
    vertices = np.vstack((stl_mesh.v0, stl_mesh.v1, stl_mesh.v2))
    unique_vertices = np.unique(vertices, axis=0)
    
    # Downsample if required
    if downsample_rate is not None and 0 < downsample_rate < 1:
        unique_vertices = shuffle(unique_vertices, random_state=42)[:int(downsample_rate * len(unique_vertices))]
    
    # Build a KD-tree from the unique (and possibly downsampled) vertices
    kd_tree = cKDTree(unique_vertices)
    
    return kd_tree


def visualize_trajectories(trajectories, ref_positions, velocities=None):
    """
    Visualize 3D trajectories with optional velocity vectors.
    trajectories: List of trajectories (each Nx3 array).
    ref_positions: Nx3 array for reference trajectory.
    velocities: Optional list of velocities corresponding to trajectories.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot reference trajectory
    ref_positions = np.array(ref_positions)
    ax.plot(ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2], label="Reference", color="blue")
    
    # Plot sampled trajectories
    for idx, trajectory in enumerate(trajectories):
        traj = np.array(trajectory)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.7, label=f"Trajectory {idx+1}")
        if velocities is not None:
            velocity = velocities[idx]
            ax.quiver(
                traj[:, 0], traj[:, 1], traj[:, 2],
                velocity[:, 0], velocity[:, 1], velocity[:, 2],
                length=0.1, normalize=True, alpha=0.5
            )
    
    ax.set_title("Trajectory Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()

def generate_bspline_trajectory(control_points, total_time=1.0, dt=0.1):
    """
    Generate a cubic B-spline trajectory, including positions, velocities, and accelerations.

    Parameters:
        control_points: (3,3) array of control points (3D).
        total_time: Total time of the trajectory (default: 1.0 seconds).
        dt: Time step for discretization.

    Returns:
        t: Time array.
        positions: Nx3 array of positions along the trajectory.
        velocities: Nx3 array of velocities along the trajectory.
        accelerations: Nx3 array of accelerations along the trajectory.
    """
    knots = [0, 0, 0, 1, 1, 1]  # Clamped knot vector for cubic B-spline
    control_points = np.array(control_points)
    t = np.arange(0, total_time + dt, dt)
    t_normalized = t / total_time

    # Create BSpline objects for x, y, z
    cx = BSpline(knots, control_points[:, 0], k=3)
    cy = BSpline(knots, control_points[:, 1], k=3)
    cz = BSpline(knots, control_points[:, 2], k=3)

    # Compute positions, velocities, and accelerations
    positions = np.column_stack([cx(t_normalized), cy(t_normalized), cz(t_normalized)])
    velocities = np.column_stack([cx.derivative(1)(t_normalized), cy.derivative(1)(t_normalized), cz.derivative(1)(t_normalized)]) / total_time
    accelerations = np.column_stack([cx.derivative(2)(t_normalized), cy.derivative(2)(t_normalized), cz.derivative(2)(t_normalized)]) / (total_time**2)

    return t, positions, velocities, accelerations
