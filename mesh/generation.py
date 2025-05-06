import numpy as np
from scipy.spatial import Delaunay

def generate_mesh(n):
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    points = []
    point_index = {}

    def bary_to_cart(l1, l2, l3):
        return l1 * A + l2 * B + l3 * C

    for i in range(n+1):
        for j in range(n+1 - i):
            k = n - i - j
            l1, l2, l3 = i/n, j/n, k/n
            pt = bary_to_cart(l1, l2, l3)
            idx = len(points)
            points.append(pt)
            point_index[(i, j)] = idx

    triangles = []
    for i in range(n):
        for j in range(n - i):
            v0 = point_index[(i, j)]
            v1 = point_index[(i+1, j)]
            v2 = point_index[(i, j+1)]
            triangles.append([v0, v1, v2])
            if i + j < n - 1:
                v3 = point_index[(i+1, j+1)]
                triangles.append([v1, v3, v2])

    return np.array(points), np.array(triangles)

def generate_perturbed_mesh(n, shift_factor=0.1):
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])

    def bary_to_cart(l1, l2, l3):
        return l1*A + l2*B + l3*C

    bary_coords = []
    cart_points = []
    boundary_mask = []

    for i in range(n+1):
        for j in range(n+1 - i):
            k = n - i - j
            l1, l2, l3 = i/n, j/n, k/n
            bary_coords.append((l1, l2, l3))
            pt = bary_to_cart(l1, l2, l3)
            cart_points.append(pt)
            boundary_mask.append(l1 == 0 or l2 == 0 or l3 == 0)

    cart_points = np.array(cart_points)
    boundary_mask = np.array(boundary_mask)

    h = 1 / n
    delta = shift_factor * h

    rng = np.random.default_rng(seed=42)
    directions = {
        0: np.array([-delta, 0]),
        1: np.array([+delta, 0]),
        2: np.array([0, +delta]),
        3: np.array([0, -delta])
    }

    for i, is_boundary in enumerate(boundary_mask):
        if not is_boundary:
            direction = rng.integers(0, 4)
            cart_points[i] += directions[direction]

    tri = Delaunay(cart_points)
    triangles = tri.simplices

    return cart_points, triangles

def oriented_edges(triangles):
    edge_dict = {}
    edges = []
    edge_map = []
    for tri in triangles:
        tri_edges = [(tri[i], tri[(i+1)%3]) for i in range(3)]
        tri_oriented = []
        for a, b in tri_edges:
            e = tuple(sorted((a, b)))
            if e not in edge_dict:
                edge_dict[e] = len(edges)
                edges.append(e)
            index = edge_dict[e]
            sign = +1 if (a, b) == e else -1
            tri_oriented.append((index, sign))
        edge_map.append(tri_oriented)
    return np.array(edges), edge_map