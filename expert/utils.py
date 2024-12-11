# utils.py
import numpy as np
from scipy.interpolate import BSpline

def generate_bspline_trajectory(control_points, total_time=1.0, dt=0.1):
    """
    control_points: (3,3) array: 3 points, each [x,y,z].
    We'll define a uniform knot vector suitable for cubic B-spline with 3 control points.
    For a cubic B-spline with 3 control points, we need a clamped knot vector:
    knots = [0,0,0, 1,1,1] might be used, and param t in [0,1].
    """
    control_points = np.array(control_points)
    # Parametric domain
    t = np.arange(0, total_time+dt, dt)
    t_normalized = t/total_time

    # For cubic and 3 points, define knots:
    knots = [0,0,0,1,1,1]

    # Separate splines for x, y, z
    cx = BSpline(knots, control_points[:,0], k=3)
    cy = BSpline(knots, control_points[:,1], k=3)
    cz = BSpline(knots, control_points[:,2], k=3)

    positions = np.column_stack([cx(t_normalized), cy(t_normalized), cz(t_normalized)])
    return t, positions

def compute_velocity_acceleration(control_points, total_time=1.0, dt=0.1):
    knots = [0,0,0,1,1,1]
    control_points = np.array(control_points)
    t = np.arange(0, total_time+dt, dt)
    t_normalized = t/total_time

    cx = BSpline(knots, control_points[:,0], k=3)
    cy = BSpline(knots, control_points[:,1], k=3)
    cz = BSpline(knots, control_points[:,2], k=3)

    positions = np.column_stack([cx(t_normalized), cy(t_normalized), cz(t_normalized)])
    velocities = np.column_stack([cx.derivative(1)(t_normalized), cy.derivative(1)(t_normalized), cz.derivative(1)(t_normalized)]) / total_time
    accelerations = np.column_stack([cx.derivative(2)(t_normalized), cy.derivative(2)(t_normalized), cz.derivative(2)(t_normalized)]) / (total_time**2)

    return t, positions, velocities, accelerations
