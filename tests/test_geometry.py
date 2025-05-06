import numpy as np
from mesh.geometry import triangle_area, circumcenter

def test_triangle_area_right_triangle():
    pts = np.array([[0, 0], [1, 0], [0, 1]])
    assert np.isclose(triangle_area(pts), 0.5)

def test_triangle_area_equilateral():
    # Equilateral triangle of side length 1 has area sqrt(3)/4
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2]
    ])
    expected = np.sqrt(3) / 4
    assert np.isclose(triangle_area(pts), expected)

def test_triangle_area_colinear():
    # Colinear points should give zero area
    pts = np.array([[0, 0], [1, 1], [2, 2]])
    assert np.isclose(triangle_area(pts), 0.0)

def test_triangle_area_reversed_order():
    # Reversing the vertex order shouldn't change the (absolute) area
    pts1 = np.array([[0, 0], [1, 0], [0, 1]])
    pts2 = pts1[::-1]
    assert np.isclose(triangle_area(pts2), triangle_area(pts1))


def test_circumcenter_right_triangle():
    # Right triangle with vertices at (0,0), (2,0), (0,2) has circumcenter at (1,1)
    tri = np.array([[0, 0], [2, 0], [0, 2]])
    cc = circumcenter(tri)
    assert np.allclose(cc, [1.0, 1.0])

def test_circumcenter_equilateral():
    # Equilateral triangle of side length 1 has circumcenter at (0.5, sqrt(3)/6)
    tri = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2]
    ])
    expected = np.array([0.5, np.sqrt(3) / 6])
    cc = circumcenter(tri)
    assert np.allclose(cc, expected)

def test_circumcenter_permuted_vertices():
    # Permuting the input vertices shouldn't affect the result
    base = np.array([[1, 1], [3, 1], [1, 3]])
    for perm in [
        base,
        base[[1, 2, 0]],
        base[[2, 0, 1]],
        base[::-1],
    ]:
        cc = circumcenter(perm)
        # This 1-1-√2 right triangle has circumcenter at (2,2)
        assert np.allclose(cc, [2.0, 2.0])

def run_all_tests_geometry():
    test_triangle_area_right_triangle()
    test_triangle_area_equilateral()
    test_triangle_area_colinear()
    test_triangle_area_reversed_order()
    test_circumcenter_right_triangle()
    test_circumcenter_equilateral()
    test_circumcenter_permuted_vertices()
    print("All Geometry tests passed.")