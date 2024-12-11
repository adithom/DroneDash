# visualizer.py
import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory_distribution(samples, filename="output/trajectories_distribution.png"):
    # samples is a list of (cost, control_points)
    costs = [s[0] for s in samples]
    # histogram of costs
    plt.figure()
    plt.hist(costs, bins=50)
    plt.xlabel("Cost")
    plt.ylabel("Frequency")
    plt.title("Distribution of Sampled Trajectory Costs")
    plt.savefig(filename)
    plt.close()

def plot_samples_3d(samples, filename="output/trajectories_3d.png"):
    # Plot the final control points distribution in 3D
    cpoints = np.array([s[1].flatten() for s in samples])
    # Just plot first control point positions as example
    p0 = cpoints[:,0:3]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p0[:,0], p0[:,1], p0[:,2], alpha=0.5)
    ax.set_title("Distribution of First Control Point")
    plt.savefig(filename)
    plt.close()
