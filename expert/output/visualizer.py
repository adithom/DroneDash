import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colormaps

# Load the CSV data for sampled trajectories
csv_file = "samples_by_time.csv"  # Replace with your actual file path
data = pd.read_csv(csv_file)

# Load the reference trajectory from a separate CSV file
ref_csv_file = "reference_trajectory.csv"  # Replace with your actual file path
ref_data = pd.read_csv(ref_csv_file)

# Extract unique time intervals and samples
time_intervals = data["time"].unique()
unique_samples = data["sample"].unique()

# Create a colormap for samples
colormap = colormaps.get_cmap("viridis")
sample_colors = {sample: colormap(i / len(unique_samples)) for i, sample in enumerate(unique_samples)}

# Determine the global axis limits based on both sampled trajectories and the reference trajectory
all_points = pd.concat([
    data[["cp1x", "cp1y", "cp1z"]].stack(),
    data[["cp2x", "cp2y", "cp2z"]].stack(),
    data[["cp3x", "cp3y", "cp3z"]].stack(),
    data[["cp4x", "cp4y", "cp4z"]].stack(),
    ref_data[["pos_x", "pos_y", "pos_z"]].stack(),
])

x_limits = (all_points.min(), all_points.max())
y_limits = (all_points.min(), all_points.max())
z_limits = (all_points.min(), all_points.max())

# Plot all trajectories with reference trajectory
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for time in time_intervals:
    subset = data[data["time"] == time]
    for _, row in subset.iterrows():
        control_points = [
            (row["cp1x"], row["cp1y"], row["cp1z"]),
            (row["cp2x"], row["cp2y"], row["cp2z"]),
            (row["cp3x"], row["cp3y"], row["cp3z"]),
            (row["cp4x"], row["cp4y"], row["cp4z"]),
        ]
        trajectory = list(zip(*control_points))  # Convert to list for reuse
        ax.plot(*trajectory, color=sample_colors[row["sample"]], alpha=0.6)

# Add reference trajectory from the separate CSV file
ref_points = ref_data[["pos_x", "pos_y", "pos_z"]].values.T  # Use correct column names
ax.plot(*ref_points, color='blue', linewidth=2, label='Reference Trajectory')

# Set consistent axis limits
ax.set_xlim(*x_limits)
ax.set_ylim(*y_limits)
ax.set_zlim(*z_limits)

# Configure plot
ax.set_title("Sampled Trajectories Across Time Intervals")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

# Save the plot with all trajectories
plt.savefig("all_trajectories.png")
plt.show()

# Plot only the reference trajectory
fig_ref = plt.figure(figsize=(8, 6))
ax_ref = fig_ref.add_subplot(111, projection='3d')

# Plot reference trajectory only
ax_ref.plot(*ref_points, color='blue', linewidth=2, label='Reference Trajectory')

# Set consistent axis limits
ax_ref.set_xlim(*x_limits)
ax_ref.set_ylim(*y_limits)
ax_ref.set_zlim(*z_limits)

# Configure plot
ax_ref.set_title("Reference Trajectory")
ax_ref.set_xlabel("X")
ax_ref.set_ylabel("Y")
ax_ref.set_zlabel("Z")
ax_ref.legend()

# Save the plot with only the reference trajectory
plt.savefig("reference_trajectory.png")
plt.show()
