import numpy as np
from dec.integrators import integrate_edge, integrate_over_triangle, compute_1_cochain, compute_2_cochain

def test_integrate_edge_constant_x():
    # Edge from (0,0) to (1,1): F = [1,0] => integral = Δx = 1
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    F = lambda x, y: np.array([1.0, 0.0])
    val = integrate_edge(F, a, b)
    assert np.isclose(val, b[0] - a[0])

def test_integrate_edge_constant_y():
    # Edge from (0,0) to (1,1): F = [0,1] => integral = Δy = 1
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    F = lambda x, y: np.array([0.0, 1.0])
    val = integrate_edge(F, a, b)
    assert np.isclose(val, b[1] - a[1])

def test_integrate_over_triangle_constant():
    # Triangle with area 0.5: f = 1 => integral = area
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    f = lambda x, y: 1.0
    val = integrate_over_triangle(tri, f)
    assert np.isclose(val, 0.5)

def test_integrate_over_triangle_linear_x():
    # ∫ x dA over triangle = area * centroid_x
    tri = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    f = lambda x, y: x
    area = 0.5 * 2.0 * 1.0
    centroid_x = (0.0 + 2.0 + 0.0) / 3.0
    expected = area * centroid_x
    val = integrate_over_triangle(tri, f)
    assert np.isclose(val, expected, rtol=1e-6)

def test_compute_1_cochain_simple():
    # F = [1,0] on edges of a right triangle: edges oriented as [(0,1),(1,2),(2,0)]
    points = np.array([[0.0, 0.0],
                       [1.0, 0.0],
                       [1.0, 1.0]])
    edges = [(0,1), (1,2), (2,0)]
    vec = lambda x, y: np.array([1.0, 0.0])

    vals = compute_1_cochain(points, edges, vec)
    # Integral along 0→1 is +1, along 1→2 is  0, along 2→0 is -1
    expected = np.array([ 1.0,  0.0, -1.0])
    assert np.allclose(vals, expected, atol=1e-8)

def test_compute_2_cochain_simple_linear_sum():
    # ∫ (x+y) dA over triangle = area * (centroid_x + centroid_y)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    triangles = np.array([[0, 1, 2]])
    func = lambda x, y: x + y
    area = 0.5
    centroid = np.array([0.0 + 1.0 + 0.0, 0.0 + 0.0 + 1.0]) / 3.0
    expected = area * (centroid[0] + centroid[1])
    vals = compute_2_cochain(points, triangles, func)
    assert np.isclose(vals[0], expected, rtol=1e-6)

def run_all_tests_integrators():
    test_integrate_edge_constant_x()
    test_integrate_edge_constant_y()
    test_integrate_over_triangle_constant()
    test_integrate_over_triangle_linear_x()
    test_compute_1_cochain_simple()
    test_compute_2_cochain_simple_linear_sum()
    print("All Integrator tests passed.")