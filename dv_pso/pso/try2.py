import json
import math
import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import LineString
from scipy import interpolate

import pso  # <- This imports the improved pso.py below
from utils import plot_lines, get_closet_points

def main():
    # PARAMETERS
    N_SECTORS = 40
    N_PARTICLES = 60
    N_ITERATIONS = 75
    W = -0.2256
    CP = -0.1564
    CG = 3.8876
    PLOT = True

    # Read tracks from json file
    with open('data/tracks.json') as file:
        json_data = json.load(file)

    track_layout = json_data['test_track']['layout']
    track_width = json_data['test_track']['width']

    # Compute inner and outer track borders using shapely
    center_line = LineString(track_layout)
    inside_line = center_line.parallel_offset(track_width / 2, 'left')
    outside_line = center_line.parallel_offset(track_width / 2, 'right')

    # Optional: Plot raw points & lines
    if PLOT:
        plt.title("Track Layout Points")
        for p in track_layout:
            plt.plot(p[0], p[1], 'r.')
        plt.show()

        plt.title("Track Layout with Borders")
        plot_lines([outside_line, inside_line])
        plt.show()

    # Define sectors' inside/outside “anchor” points
    inside_points, outside_points = define_sectors(center_line, inside_line, outside_line, N_SECTORS)

    # Plot all sector lines
    if PLOT:
        plt.title("Sectors")
        for i in range(N_SECTORS):
            plt.plot([inside_points[i][0], outside_points[i][0]],
                     [inside_points[i][1], outside_points[i][1]])
        plot_lines([outside_line, inside_line])
        plt.show()

    # Define boundaries for PSO ([0, distance between inside and outside points])
    boundaries = [np.linalg.norm(inside_points[i] - outside_points[i]) for i in range(N_SECTORS)]
    print("Boundaries for each sector:", boundaries)

    # --- COST FUNCTION --- #
    def myCostFunc(sectors):
        """
        Cost function = lap time + optional smoothing penalty
        """
        return get_lap_time(
            racing_line=sectors_to_racing_line(sectors, inside_points, outside_points),
            smoothness_penalty=False   # Change to True if you want a smoothing penalty
        )

    # --- RUN PSO --- #
    global_solution, gs_eval, gs_history, gs_eval_history = pso.optimize(
        cost_func=myCostFunc,
        n_dimensions=N_SECTORS,
        boundaries=boundaries,
        n_particles=N_PARTICLES,
        n_iterations=N_ITERATIONS,
        w=W,
        cp=CP,
        cg=CG,
        verbose=True
    )

    # Evaluate best solution
    best_line = sectors_to_racing_line(global_solution, inside_points, outside_points)
    final_lap_time, v, x, y = get_lap_time(best_line, return_all=True)
    print("Final Lap Time:", final_lap_time)

    # Optional: Plot the convergence of solutions
    if PLOT:
        plt.title("Racing Line Evolution")
        plt.ion()
        for i in range(len(gs_history)):
            # Convert each iteration's global best to a racing line
            tmp_line = sectors_to_racing_line(gs_history[i], inside_points, outside_points)
            lth, vh, xh, yh = get_lap_time(tmp_line, return_all=True)

            plt.scatter(xh, yh, marker='.', c=vh, cmap='RdYlGn', alpha=0.8)
            plot_lines([outside_line, inside_line])
            plt.draw()
            plt.pause(0.001)
            plt.clf()
        plt.ioff()

        # Final result
        plt.title("Final racing line")
        rl = np.array(best_line)
        plt.plot(rl[:, 0], rl[:, 1], 'r-')
        plt.scatter(x, y, marker='.', c=v, cmap='RdYlGn')
        for i in range(N_SECTORS):
            plt.plot([inside_points[i][0], outside_points[i][0]],
                     [inside_points[i][1], outside_points[i][1]])
        plot_lines([outside_line, inside_line])
        plt.show()

        plt.title("Global solution history")
        plt.ylabel("Lap time (s)")
        plt.xlabel("Iteration")
        plt.plot(gs_eval_history)
        plt.show()

def sectors_to_racing_line(sectors, inside_points, outside_points):
    """
    Converts the sector values (scalar distances from the inside to the outside)
    into XY coordinates on the track.
    """
    racing_line = []
    for i in range(len(sectors)):
        x1, y1 = inside_points[i]
        x2, y2 = outside_points[i]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)

        # Avoid division by zero
        if length == 0:
            xp, yp = x1, y1
        else:
            t = sectors[i] / length  # fraction [0..1]
            xp = x1 + t * dx
            yp = y1 + t * dy

        racing_line.append([xp, yp])
    return racing_line

def get_lap_time(racing_line, return_all=False, smoothness_penalty=False):
    """
    Computes the lap time using a spline approximation to measure distance,
    curvature, and hence speed along the path.

    If smoothness_penalty=True, we add a small penalty based on how 'abrupt'
    the sector changes are (discouraging big jumps between consecutive sectors).
    """

    rl = np.array(racing_line)

    # -- Quick guard: If the user accidentally passes too few points --
    if len(rl) < 2:
        if return_all:
            return 9999.0, [], [], []
        return 9999.0

    # 1) Interpolate the racing line with fewer points for speed
    #    Try a smaller resolution than 1000 to reduce computation
    n_spline_points = 300
    tck, _ = interpolate.splprep([rl[:, 0], rl[:, 1]], s=0, per=0)
    svals = np.linspace(0, 1, n_spline_points)
    x, y = interpolate.splev(svals, tck)

    # 2) Compute first and second derivatives (for curvature)
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # curvature = |x' * y'' - y' * x''| / (x'^2 + y'^2)^(3/2)
    denom = (dx**2 + dy**2) ** 1.5
    # To avoid divide-by-zero, add a small epsilon where denom is zero
    denom[denom == 0] = 1e-9
    curvature = np.abs(dx * d2y - dy * d2x) / denom

    # 3) Compute speed at each point
    us = 0.13
    max_speed = 10.0
    # radius = 1/curvature
    with np.errstate(divide='ignore'):
        radius = 1.0 / curvature
    radius[np.isinf(radius)] = 1e9  # effectively a very large radius

    # v_i = min(max_speed, sqrt(mu*g*radius_i))
    g = 9.81
    v = np.minimum(max_speed, np.sqrt(us * radius * g))

    # 4) Compute total distance along the track
    # Distances between consecutive spline points
    dx2 = np.diff(x)
    dy2 = np.diff(y)
    seg_distances = np.sqrt(dx2**2 + dy2**2)

    # 5) Approximate time = sum of (segment_length / speed_of_segment)
    # For speed, we can take v[i] as speed on the segment [i -> i+1].
    # One option: v_seg = (v[i] + v[i+1]) / 2.0, or simply v[i].
    # We'll do a simple approach: v[i] for the i-th segment.
    # This means we need seg_distances.size == v.size - 1
    v_seg = v[:-1]  # same length as seg_distances
    lap_time = np.sum(seg_distances / v_seg)

    # 6) Optional smoothing penalty
    penalty = 0.0
    if smoothness_penalty:
        # A typical penalty is the sum of the squared differences
        # between consecutive sector “positions” to discourage huge jumps
        # across sectors.  This example just uses the raw 'rl' points in X/Y.
        diffs = np.sqrt(np.diff(rl[:, 0])**2 + np.diff(rl[:, 1])**2)
        penalty = 0.01 * np.sum(diffs**2)  # scale factor is problem-dependent

    total_cost = lap_time + penalty

    if return_all:
        return total_cost, v, x, y
    return total_cost

def define_sectors(center_line, inside_line, outside_line, n_sectors):
    """
    Defines the 'inside' and 'outside' points for each sector by interpolating
    the center line and then finding the closest points on the inside/outside.

    Returns
    -------
    inside_points : ndarray of shape (n_sectors, 2)
    outside_points : ndarray of shape (n_sectors, 2)
    """
    # Interpolate equidistant points along the center line
    distances = np.linspace(0, center_line.length, n_sectors)
    center_pts_temp = [center_line.interpolate(d) for d in distances]
    center_pts = np.array([[p.x, p.y] for p in center_pts_temp])

    # We want the track to be closed, so we’ll make the last sector the same as the first
    # if it’s truly a loop
    if (center_pts[0] != center_pts[-1]).any():
        center_pts = np.vstack([center_pts, center_pts[0]])

    # For inside/outside lines, we sample once at high resolution
    # (doing this once is cheaper than repeated calls).
    N_SAMPLES = 500  # fewer samples to speed up
    inside_dists = np.linspace(0, inside_line.length, N_SAMPLES)
    outside_dists = np.linspace(0, outside_line.length, N_SAMPLES)
    inside_border = np.array([[inside_line.interpolate(d).x, inside_line.interpolate(d).y]
                              for d in inside_dists])
    outside_border = np.array([[outside_line.interpolate(d).x, outside_line.interpolate(d).y]
                               for d in outside_dists])

    # For each center point, get the nearest inside/outside point
    inside_points = np.array([get_closet_points(cp, inside_border) for cp in center_pts])
    outside_points = np.array([get_closet_points(cp, outside_border) for cp in center_pts])

    return inside_points, outside_points

if __name__ == "__main__":
    main()
