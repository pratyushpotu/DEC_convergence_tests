import numpy as np
from scipy.integrate import quad

def integrate_edge(F, a, b):
    def integrand(s):
        p = a + s * (b - a)
        t = b - a
        t_unit = t / np.linalg.norm(t)
        return np.dot(F(p[0], p[1]), t_unit)
    val, _ = quad(integrand, 0.0, 1.0, epsabs=1e-8, epsrel=1e-8)
    return val * np.linalg.norm(b - a)

def integrate_over_triangle(triangle_coords, func):
    weights = np.array([
        0.225,
        0.132394152788, 0.132394152788, 0.132394152788,
        0.125939180544, 0.125939180544, 0.125939180544
    ])
    bary_coords = np.array([
        [1/3, 1/3, 1/3],
        [0.0597158717898, 0.470142064105, 0.470142064105],
        [0.470142064105, 0.0597158717898, 0.470142064105],
        [0.470142064105, 0.470142064105, 0.0597158717898],
        [0.797426985353, 0.101286507323, 0.101286507323],
        [0.101286507323, 0.797426985353, 0.101286507323],
        [0.101286507323, 0.101286507323, 0.797426985353]
    ])

    A, B, C = triangle_coords
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
    integral = 0.0
    for w, (l1, l2, l3) in zip(weights, bary_coords):
        p = l1*A + l2*B + l3*C
        x, y = p
        integral += w * func(x, y)

    return area * integral

def compute_1_cochain(points, edges, vec):
    return np.array([
        integrate_edge(vec, points[a], points[b])
        for a, b in edges
    ])

def compute_2_cochain(points, triangles, func):
    vals = np.zeros(len(triangles))
    for i, tri in enumerate(triangles):
        tri_coords = points[tri]
        integral = integrate_over_triangle(tri_coords, func)
        vals[i] = integral
    return vals