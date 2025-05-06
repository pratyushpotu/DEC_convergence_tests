import numpy as np
from mesh.generation import generate_mesh, oriented_edges
from dec.hodge import build_H0, build_H1, build_H2

def test_hodge_shapes():
    points, triangles = generate_mesh(2)
    edges, edge_map = oriented_edges(triangles)

    H0 = build_H0(points, edges, triangles, edge_map)
    H1 = build_H1(points, edges, triangles, edge_map)
    H2 = build_H2(points, triangles)

    assert H0.shape[0] == len(points)
    assert H1.shape[0] == len(edges)
    assert H2.shape[0] == len(triangles)

def test_hodge_positivity():
    points, triangles = generate_mesh(2)
    edges, edge_map = oriented_edges(triangles)

    H0 = build_H0(points, edges, triangles, edge_map)
    H1 = build_H1(points, edges, triangles, edge_map)
    H2 = build_H2(points, triangles)

    assert np.all(H0.diagonal() > 0)
    assert np.all(H1.diagonal() > 0)
    assert np.all(H2.diagonal() > 0)

def test_hodge_values_n2():
    points, triangles = generate_mesh(2)
    edges, edge_map = oriented_edges(triangles)

    H0 = build_H0(points, edges, triangles, edge_map)
    H1 = build_H1(points, edges, triangles, edge_map)
    H2 = build_H2(points, triangles)

    # Check expected values (pre-computed for n=2 symmetric mesh)
    expected_H0 = np.array([
        0.03608439,  # corner C
        0.10825318,  # mid-edge BC
        0.03608439,  # corner B
        0.10825318,  # mid-edge AC
        0.10825318,  # mid-edge AB
        0.03608439   # corner A
    ])
    expected_H1 = np.array([
        0.28867513,  # edge (0,1)
        0.57735027,  # edge (0,3)
        0.28867513,  # edge (1,3)
        0.57735027,  # edge (1,4)
        0.57735027,  # edge (2,4)
        0.28867513,  # edge (1,2)
        0.28867513,  # edge (3,5)
        0.28867513,  # edge (4,5)
        0.28867513   # edge (3,4)
    ])
    expected_H2 = np.full(len(triangles), 16/np.sqrt(3)) 

    # 5) Compare diagonals
    np.testing.assert_allclose(H0.diagonal(), expected_H0, rtol=1e-7)
    np.testing.assert_allclose(H1.diagonal(), expected_H1, rtol=1e-7)
    np.testing.assert_allclose(H2.diagonal(), expected_H2, rtol=1e-7)


def run_all_tests_hodge():
    test_hodge_shapes()
    test_hodge_positivity()
    test_hodge_values_n2()
    print("All Hodge star tests passed.")