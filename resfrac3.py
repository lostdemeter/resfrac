import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from mpmath import *
from sympy import primerange, primepi, isprime, mobius
from math import exp, log, pi, sqrt
import time

mp.dps = 20

def riemann_R(x, K=50):
    s = mpf(0)
    for n in range(1, K+1):
        mu = mobius(n)
        if mu == 0:
            continue
        s += mu / n * li(x ** (1/n))
    return float(s)

def get_gammas_dynamic(num_zeros):
    return np.array([float(im(zetazero(k))) for k in range(1, num_zeros + 1)])

def segmented_pre_sieve(start_n, end_n, B):
    small_primes = list(primerange(2, B + 1))
    length = end_n - start_n + 1
    is_candidate = [True] * length
    for p in small_primes:
        start_multiple = max(p * p, ((start_n + p - 1) // p) * p)
        if start_multiple > end_n:
            continue
        idx = start_multiple - start_n
        for i in range(idx, length, p):
            is_candidate[i] = False
    return [start_n + i for i in range(length) if is_candidate[i]]

def compute_spectral_scores(candidates, gammas, h):
    log_ns = np.log(candidates)
    osc_plus = np.zeros(len(candidates), dtype=complex)
    osc_minus = np.zeros(len(candidates), dtype=complex)
    for gamma in gammas:
        taper = exp(-0.5 * h**2 * gamma**2)
        shift_plus = (0.5 + 1j * gamma) * h
        shift_minus = (0.5 + 1j * gamma) * (-h)
        phases_plus = np.exp(1j * gamma * log_ns + shift_plus)
        phases_minus = np.exp(1j * gamma * log_ns + shift_minus)
        osc_plus += taper * phases_plus / (0.5 + 1j * gamma)
        osc_minus += taper * phases_minus / (0.5 + 1j * gamma)
    psi_plus = np.array(candidates) * exp(h) - 2 * np.real(osc_plus)
    psi_minus = np.array(candidates) * exp(-h) - 2 * np.real(osc_minus)
    logn_arr = np.log(candidates)
    scores = (psi_plus - psi_minus) / (2 * h * np.array(candidates) * logn_arr)
    return scores

def chudnovsky_like_sieve(N, T=50, K=50, epsilon=1.2):
    gammas = get_gammas_dynamic(T)
    B = int(sqrt(N)) + 1
    mid = N / 2
    h = 0.05 / log(mid)
    approx = riemann_R(N, K)
    M = int(ceil(epsilon * approx))
    candidates = segmented_pre_sieve(2, N, B)
    scores = compute_spectral_scores(candidates, gammas, h)
    mean_s = np.mean(scores)
    sigma_s = max(np.std(scores), 0.01)
    z_scores = (scores - mean_s) / sigma_s
    top_idx = np.argsort(-z_scores)[:M]
    top_candidates = np.array(candidates)[top_idx]
    primes = [int(c) for c in top_candidates if isprime(int(c))]
    return sorted(primes)

class ResonantSolver:
    def __init__(self, phi=(1 + np.sqrt(5)) / 2, alpha=0.05, max_iters=50, borwein_terms=12):
        self.phi = phi
        self.alpha = alpha
        self.max_iters = max_iters
        self.borwein_terms = borwein_terms
        self.lengths = []
    
    def _n_points(self, graph, dist_matrix=None):
        if hasattr(graph, 'coords') and graph.coords is not None:
            return graph.coords.shape[0]
        if hasattr(graph, 'vars'):
            return graph.vars
        if dist_matrix is not None:
            return dist_matrix.shape[0]
        raise ValueError('Cannot determine problem size')

    def borwein_weights(self, dist_matrix):
        n_local = dist_matrix.shape[0]
        smoothed = dist_matrix.copy().astype(float)
        for i in range(n_local):
            for j in range(n_local):
                if i == j:
                    continue
                k = np.arange(1, self.borwein_terms + 1)
                decay = np.sum(np.sinc(k * dist_matrix[i, j] / np.pi) ** 2)
                smoothed[i, j] *= 0.5 + 0.5 * (decay / self.borwein_terms)
        return smoothed
    
    def phi_greedy(self, graph, dist_matrix):
        n_local = self._n_points(graph, dist_matrix)
        tour = [0]
        visited = np.zeros(n_local, bool)
        visited[0] = True
        current = 0
        for _ in range(n_local - 1):
            dists = dist_matrix[current].copy()
            dists[visited] = np.inf
            if hasattr(graph, 'coords') and graph.coords is not None:
                angles = np.arctan2(graph.coords[:, 1] - graph.coords[current, 1],
                                    graph.coords[:, 0] - graph.coords[current, 0])
                bias = np.exp(-np.abs(angles - (np.pi / self.phi)) / 0.5)
            else:
                bias = np.ones(n_local)
            scores = dists / (1 + bias)
            next_node = int(np.argmin(scores))
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
        tour.append(tour[0])
        return np.array(tour, dtype=int)
    
    def two_opt(self, tour, dist_matrix):
        tour = tour.copy()
        improved = True
        while improved:
            improved = False
            for i in range(1, len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    if j - i == 1:
                        continue
                    k = (j + 1) % len(tour)
                    old1 = dist_matrix[tour[i - 1], tour[i]]
                    old2 = dist_matrix[tour[j], tour[k]]
                    new1 = dist_matrix[tour[i - 1], tour[j]]
                    new2 = dist_matrix[tour[i], tour[k]]
                    if new1 + new2 < old1 + old2:
                        tour[i:j + 1] = tour[j:i - 1:-1]
                        improved = True
        return tour
    
    def align_to_y(self, points):
        idx_top = np.argmax(points[:, 1])
        theta0 = np.arctan2(points[idx_top, 1], points[idx_top, 0])
        cos, sin = np.cos(-theta0), np.sin(-theta0)
        rot = np.array([[cos, sin], [-sin, cos]])
        return points @ rot
    
    def flip_x(self, points):
        return np.column_stack((-points[:, 0], points[:, 1]))
    
    def get_mids(self, tour_coords):
        return (tour_coords[:-1] + tour_coords[1:]) / 2
    
    def dual_improve(self, tour, graph, dist_matrix):
        if hasattr(graph, 'coords') and graph.coords is not None:
            tour_coords = graph.coords[tour]
            mids = self.get_mids(tour_coords)
            aligned_graph = self.align_to_y(graph.coords)
            aligned_mids = self.align_to_y(mids)
            flipped_graph = self.flip_x(aligned_graph)
            flipped_mids = self.flip_x(aligned_mids)
            cost = np.linalg.norm(flipped_mids[:, np.newaxis] - flipped_graph[np.newaxis, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost)
            n_local = cost.shape[1]
            city_pos = np.full(n_local, np.nan)
            for r, c in zip(row_ind, col_ind):
                city_pos[c] = r
            sort_idx = np.argsort(city_pos)
            candidate_open = self.two_opt(sort_idx.copy(), dist_matrix)
            candidate_tour = np.append(candidate_open, candidate_open[0])
            cand_score = self.score_solution(candidate_tour, graph, dist_matrix)
            curr_score = self.score_solution(tour, graph, dist_matrix)
            if cand_score < curr_score:
                return candidate_tour, cand_score
        return tour, self.score_solution(tour, graph, dist_matrix)
    
    def score_solution(self, solution, graph, dist_matrix=None):
        problem_type = getattr(graph, 'type', 'tsp')
        if problem_type == 'tsp':
            return np.sum(dist_matrix[solution[:-1], solution[1:]]) + dist_matrix[solution[-1], solution[0]]
        elif problem_type == 'sat_3':
            unsat = 0
            for clause in graph.clauses:
                lits = [(solution[var] ^ neg) for var, neg in clause]  # True if satisfied
                if not any(lits):
                    unsat += 1
            return unsat
        elif problem_type == 'prime':
            return len(solution)  # Number of primes found (maximize, but used as is)
        return 0
    
    def _solve_tsp(self, graph):
        dist_matrix = np.linalg.norm(graph.coords[:, np.newaxis] - graph.coords[np.newaxis, :], axis=2)
        tour = self.phi_greedy(graph, dist_matrix)
        tour_open = self.two_opt(tour[:-1], dist_matrix)
        tour = np.append(tour_open, tour_open[0])
        self.lengths = [self.score_solution(tour, graph, dist_matrix)]
        for it in range(self.max_iters):
            if it % 2 == 0:
                b_dist = self.borwein_weights(dist_matrix)
                tour_temp = self.phi_greedy(graph, b_dist)
                tour_open = self.two_opt(tour_temp[:-1], b_dist)
                tour = np.append(tour_open, tour_open[0])
            tour, new_score = self.dual_improve(tour, graph, dist_matrix)
            self.lengths.append(new_score)
            if abs(new_score - self.lengths[-2]) < 1e-4:
                break
        return tour, self.lengths[-1], dist_matrix
    
    def _solve_sat(self, graph):
        n_vars = graph.vars
        rng = np.random.default_rng(0)
        assign = rng.integers(0, 2, size=n_vars)
        def sat_score(a):
            return self.score_solution(a, graph)
        best = assign.copy()
        best_score = sat_score(best)
        self.lengths = [best_score]
        
        # Precompute clause barycenters as mids on the unit circle
        angles_vars = 2 * np.pi * np.arange(n_vars) / n_vars
        clause_mids = []
        for clause in graph.clauses:
            weights = np.zeros(n_vars)
            for var, neg in clause:
                weights[var] += (1.0 if not neg else -1.0)
            # Weighted mean direction via vector sum to avoid angle wrap
            vx = np.sum(weights * np.cos(angles_vars))
            vy = np.sum(weights * np.sin(angles_vars))
            norm = np.hypot(vx, vy) + 1e-12
            clause_mids.append([vx / norm, vy / norm])
        clause_mids = np.array(clause_mids) if len(clause_mids) > 0 else np.zeros((0,2))
        
        for it in range(self.max_iters * 10):
            improved = False
            # Local greedy flips
            for v in range(n_vars):
                assign[v] ^= 1
                s = sat_score(assign)
                if s <= best_score:
                    best_score = s
                    best = assign.copy()
                    improved = True
                else:
                    assign[v] ^= 1
            self.lengths.append(best_score)
            
            # Dual jump every 5 iterations
            if it % 5 == 0 and len(clause_mids) > 0:
                # Map assignment to circle: 0 -> angle 0; 1 -> angle pi
                assign_angles = np.where(best == 0, 0.0, np.pi)
                assign_coords = np.column_stack((np.cos(assign_angles), np.sin(assign_angles)))
                # Build mids set (truncate/pad to n_vars)
                if clause_mids.shape[0] < n_vars:
                    reps = int(np.ceil(n_vars / clause_mids.shape[0]))
                    mids_use = np.vstack([clause_mids] * reps)[:n_vars]
                else:
                    mids_use = clause_mids[:n_vars]
                aligned_assign = self.align_to_y(assign_coords)
                aligned_mids = self.align_to_y(mids_use)
                flipped_assign = self.flip_x(aligned_assign)
                flipped_mids = self.flip_x(aligned_mids)
                cost = np.linalg.norm(flipped_mids[:, np.newaxis] - flipped_assign[np.newaxis, :], axis=2)
                row_ind, col_ind = linear_sum_assignment(cost)
                # Propose non-local flips based on assignment-mismatch
                candidate = best.copy()
                for r, c in zip(row_ind, col_ind):
                    # If matched mid direction is far from current bit (use dot product threshold)
                    v_mid = flipped_mids[r]
                    v_var = flipped_assign[c]
                    if v_mid @ v_var < np.cos(np.pi / self.phi):
                        candidate[c] ^= 1
                cand_score = sat_score(candidate)
                if cand_score <= best_score:
                    best = candidate
                    best_score = cand_score
                    improved = True
            
            if not improved:
                break
        return best, best_score
    
    def _solve_prime(self, graph):
        primes = chudnovsky_like_sieve(graph.N)
        self.lengths = [len(primes)]
        return primes, len(primes)
    
    def solve(self, graph, dist_matrix=None):
        problem_type = getattr(graph, 'type', 'tsp')
        if problem_type == 'tsp':
            return self._solve_tsp(graph)
        elif problem_type == 'sat_3':
            return self._solve_sat(graph)
        elif problem_type == 'prime':
            return self._solve_prime(graph)
        else:
            raise NotImplementedError(f'Unknown problem type: {problem_type}')
    
    def invariant(self, graph, solution):
        if hasattr(graph, 'coords') and graph.coords is not None:
            n_pts = graph.coords.shape[0]
            k = min(4, n_pts)
            dmat = KDTree(graph.coords).query(graph.coords, k=k)[0][:, 1:]
            d = np.log(np.mean(dmat)) / np.log(2)
        else:
            d = 1.0
        gaps = self._get_gaps(solution, graph)
        n_bins_base = graph.coords.shape[0] if hasattr(graph, 'coords') and graph.coords is not None else (len(solution) if hasattr(solution, '__len__') else 8)
        bins = max(1, min(8, n_bins_base // 2))
        hist, _ = np.histogram(gaps, bins=bins)
        probs = hist / hist.sum() if hist.sum() else np.ones_like(hist) / len(hist)
        H = -np.sum(probs * np.log2(probs + 1e-12))
        return d + H / np.log(self.phi)
    
    def _get_gaps(self, solution, graph):
        problem_type = getattr(graph, 'type', 'tsp')
        if problem_type == 'tsp':
            coords = graph.coords
            tour_coords = coords[solution]
            gaps = np.linalg.norm(np.diff(tour_coords, axis=0, append=tour_coords[0:1]), axis=1)
            return gaps
        elif problem_type == 'prime':
            return np.diff(solution)
        else:
            a = np.asarray(solution, dtype=int)
            diffs = np.abs(np.diff(a, append=a[0]))
            return diffs

# Existing SATGraph
class SATGraph:
    type = 'sat_3'
    def __init__(self, n_vars=10, n_clauses=20, clauses=None):
        self.vars = n_vars
        if clauses is not None:
            self.clauses = clauses
        else:
            self.clauses = []
            for _ in range(n_clauses):
                clause = []
                for _ in range(3):
                    var = np.random.randint(0, n_vars)
                    neg = bool(np.random.choice([True, False]))
                    clause.append((var, neg))
                self.clauses.append(clause)
        angles = 2 * np.pi * np.arange(self.vars) / self.vars if self.vars > 0 else np.array([])
        self.coords = np.column_stack((np.cos(angles), np.sin(angles))) if self.vars > 0 else np.zeros((0, 2))

# New PrimeGraph
class PrimeGraph:
    type = 'prime'
    def __init__(self, N=10000):
        self.N = N
        self.primes = chudnovsky_like_sieve(N)
        self.coords = np.column_stack((np.array(self.primes), np.zeros(len(self.primes))))

def load_dimacs(file_path):
    clauses = []
    vars = 0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('c') or line.startswith('%') or not line:
                continue
            if line.startswith('p cnf'):
                _, _, v_str, c_str = line.split()
                vars = int(v_str)
                continue
            if line == '0':
                continue
            lits = list(map(int, line.split()))
            if lits and lits[-1] == 0:
                lits = lits[:-1]
            clause = [(abs(lit)-1, lit < 0) for lit in lits]
            clauses.append(clause)
    return SATGraph(n_vars=vars, clauses=clauses)

# Example Usage
if __name__ == "__main__":
    np.random.seed(0)
    # SAT example
    #sat_g = SATGraph(n_vars=10, n_clauses=20)
    #sat_g = load_dimacs('uf50-0218.cnf')
    #solver = ResonantSolver(max_iters=50)
    #assignment, unsat = solver.solve(sat_g)
    #inv_sat = solver.invariant(sat_g, assignment)
    #print(f"SAT Assignment: {assignment}, Unsat Clauses: {unsat}, Invariant: {inv_sat:.2f}, Iters: {len(solver.lengths)-1}")
    
    # Prime example
    prime_g = PrimeGraph(N=10000)
    solver = ResonantSolver(max_iters=50)
    primes, num_primes = solver.solve(prime_g)
    inv_prime = solver.invariant(prime_g, primes)
    print(f"Primes up to {prime_g.N}: {len(primes)}, Expected: {primepi(prime_g.N)}, Invariant: {inv_prime:.2f}, Iters: {len(solver.lengths)-1}")