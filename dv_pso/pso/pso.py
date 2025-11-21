import numpy as np
import time

class ParticleSwarmOptimizer:
    def __init__(self, cost_func, n_dimensions, bounds, n_particles=100, w=0.7, c1=1.4, c2=1.4):
        """
        Vectorized Particle Swarm Optimization (PSO)
        
        Parameters
        ----------
        cost_func : callable
            Function that accepts a 1D array (particle) and returns a float (score).
        n_dimensions : int
            Number of dimensions (e.g., number of track waypoints).
        bounds : tuple or list
            Format: (min_bound, max_bound) for scalar bounds (applied to all dims)
            OR shape (2, n_dimensions) for specific bounds per dimension.
        """
        self.func = cost_func
        self.n_dim = n_dimensions
        self.n_particles = n_particles
        self.w = w      # Inertia
        self.c1 = c1    # Cognitive (Personal)
        self.c2 = c2    # Social (Global)

        # -- 1. Initialize Boundaries --
        # Handle both scalar bounds [-1, 1] and vector bounds [[-1, -1...], [1, 1...]]
        bounds = np.array(bounds)
        if bounds.ndim == 1:
            self.lb = np.full(n_dimensions, bounds[0])
            self.ub = np.full(n_dimensions, bounds[1])
        else:
            self.lb = bounds[0]
            self.ub = bounds[1]

        # -- 2. Initialize Swarm State (Vectorized) --
        # Position: Uniform random between lower and upper bounds
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(n_particles, n_dimensions))
        
        # Velocity: Initialize small (20% of search space) to prevent explosion at t=0
        v_range = (self.ub - self.lb) * 0.2
        self.V = np.random.uniform(low=-v_range, high=v_range, size=(n_particles, n_dimensions))
        
        # Personal Best (P) - Copy X to avoid reference bugs
        self.P = self.X.copy()
        self.P_fit = np.full(n_particles, np.inf)
        
        # Global Best (G)
        self.G = np.zeros(n_dimensions)
        self.G_fit = np.inf

    def optimize(self, n_iterations, verbose=False):
        """
        Runs the optimization loop.
        Returns: (best_position, best_score, history_of_positions, history_of_scores)
        """
        history = []
        eval_history = []

        if verbose:
            print(f"Starting PSO: {self.n_particles} particles, {self.n_dim} dimensions")
            start_time = time.time()

        for i in range(n_iterations):
            # -- 1. Evaluate Fitness --
            # np.apply_along_axis passes each row (particle) to cost_func
            current_fitness = np.apply_along_axis(self.func, 1, self.X)

            # -- 2. Update Personal Bests (P) --
            # Vectorized boolean mask for particles that improved
            improved_mask = current_fitness < self.P_fit
            
            self.P[improved_mask] = self.X[improved_mask]
            self.P_fit[improved_mask] = current_fitness[improved_mask]

            # -- 3. Update Global Best (G) --
            min_batch_idx = np.argmin(self.P_fit)
            min_batch_score = self.P_fit[min_batch_idx]

            if min_batch_score < self.G_fit:
                self.G_fit = min_batch_score
                self.G = self.P[min_batch_idx].copy()

            # -- 4. Update Velocity --
            # CRITICAL FIX: r1, r2 are shape (n_particles, n_dim)
            # This decouples dimensions so particles can move differently in X vs Y
            r1 = np.random.rand(self.n_particles, self.n_dim)
            r2 = np.random.rand(self.n_particles, self.n_dim)

            self.V = (self.w * self.V) + \
                     (self.c1 * r1 * (self.P - self.X)) + \
                     (self.c2 * r2 * (self.G - self.X))

            # Optional: Clamp velocity to prevent "teleporting"
            # self.V = np.clip(self.V, -(self.ub-self.lb), (self.ub-self.lb))

            # -- 5. Update Position --
            self.X = self.X + self.V

            # -- 6. Boundary Handling (Hard Constraints) --
            self.X = np.clip(self.X, self.lb, self.ub)

            # Logging
            history.append(self.G.copy())
            eval_history.append(self.G_fit)

            if verbose and (i % 10 == 0 or i == n_iterations - 1):
                print(f"Iter {i:>4} | Best Score: {self.G_fit:.6f}")

        if verbose:
            elapsed = time.time() - start_time
            print(f"Optimization finished in {elapsed:.2f}s")

        return self.G, self.G_fit, history, eval_history


# ---------------------------------------------------------
# EXAMPLE: RACING LINE OPTIMIZATION WITH SMOOTHNESS PENALTY
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # 1. Define the "Physics" (The invisible target)
    # Let's pretend the optimal line is a Sine wave (smooth)
    # Real application: This would be your track simulation code
    TARGET_LINE = np.sin(np.linspace(0, 2*np.pi, 50))

    def racing_physics_simulator(trajectory):
        """
        Simulates the car. Returns raw lap time (lower is better).
        Error = Distance from the 'perfect' line (TARGET_LINE).
        """
        # Euclidean distance to target
        raw_error = np.sum((trajectory - TARGET_LINE)**2)
        return raw_error

    # 2. Define the Cost Function wrapper with SMOOTHNESS
    def cost_func_with_smoothing(trajectory):
        # A. Get Raw Performance
        raw_score = racing_physics_simulator(trajectory)
        
        # B. Calculate Roughness (2nd Derivative)
        # Sum of squares of the 2nd derivative (acceleration of change)
        # High value = jagged line. Low value = smooth line.
        diffs = np.diff(trajectory, n=2)
        roughness = np.sum(diffs**2)
        
        # C. Combine with Penalty Weight (Lambda)
        # Increase LAMBDA if your result is too jittery
        LAMBDA = 5.0 
        return raw_score + (LAMBDA * roughness)

    # 3. Setup PSO
    print("Running PSO for Trajectory Optimization...")
    n_waypoints = 50
    track_width_bounds = [-1.5, 1.5] # Car can go 1.5m left or right of center

    optimizer = ParticleSwarmOptimizer(
        cost_func=cost_func_with_smoothing, # Pass the SMOOTH version
        n_dimensions=n_waypoints,
        bounds=track_width_bounds,
        n_particles=100,
        w=0.7, c1=1.4, c2=1.4
    )

    # 4. Run
    best_line, best_score, _, _ = optimizer.optimize(n_iterations=200, verbose=True)

    print("\nOptimization Complete.")
    print(f"Target Line (First 5): {TARGET_LINE[:5]}")
    print(f"PSO Solution (First 5): {best_line[:5]}")