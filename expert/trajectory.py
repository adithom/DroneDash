# trajectory.py
# loading reference trajectory and fitting b spline to points
import csv
import numpy as np
from scipy.interpolate import make_lsq_spline

def load_reference_trajectory(csv_filename):
    times = []
    positions = []
    velocities = []
    accelerations = []
    with open(csv_filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith('#'):
                continue
            t = float(row[0])
            px, py, pz = float(row[1]), float(row[2]), float(row[3])
            vx, vy, vz = float(row[4]), float(row[5]), float(row[6])
            ax, ay, az = float(row[7]), float(row[8]), float(row[9])

            times.append(t)
            positions.append([px, py, pz])
            velocities.append([vx, vy, vz])
            accelerations.append([ax, ay, az])

    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    return times, positions, velocities, accelerations

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r != 0 else 0
    phi = np.arctan2(y, x)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def generate_b_spline(control_points, num_points=10):
    """
    Generate a cubic B-spline from control points.
    """
    t = np.linspace(0, 1, num_points)
    degree = 3
    knots = np.concatenate(([0] * (degree + 1), np.linspace(0, 1, len(control_points) - degree), [1] * (degree + 1)))
    spline = BSpline(knots, control_points, degree)
    return spline(t)
