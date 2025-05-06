import numpy as np
from mesh.generation import generate_mesh, oriented_edges
from dec.derivatives import build_d0, build_d1

def test_d0_entries():
    # For each edge row in d0, exactly two nonzeros: -1 at start, +1 at end
    points, triangles = generate_mesh(4)
    edges, _ = oriented_edges(triangles)
    d0 = build_d0(points, edges).toarray()
    for i, (a, b) in enumerate(edges):
        row = d0[i]
        # exactly two nonzero entries at columns a and b
        nz = np.nonzero(row)[0]
        assert set(nz) == {a, b}
        assert row[a] == -1
        assert row[b] == +1

def test_d1_entries():
    # For each triangle row in d1, nonzeros correspond exactly to its edges with correct sign
    _, triangles = generate_mesh(4)
    _, edge_map = oriented_edges(triangles)
    d1 = build_d1(triangles, edge_map).toarray()
    for t_idx, tri in enumerate(edge_map):
        row = d1[t_idx]
        # Check each oriented edge appears with its sign
        for e_idx, sign in tri:
            assert row[e_idx] == sign
        # All other entries are zero
        for j in range(d1.shape[1]):
            if j not in [e for e, _ in tri]:
                assert row[j] == 0

def test_d1_d0_composition_zero():
    # d1 ⋅ d0 should be the zero matrix (boundary of boundary = 0)
    points, triangles = generate_mesh(3)
    edges, edge_map = oriented_edges(triangles)
    d0 = build_d0(points, edges)
    d1 = build_d1(triangles, edge_map)
    product = d1.dot(d0).toarray()
    assert np.allclose(product, 0)

def test_d0_constant_function_zero():
    # d0 applied to a constant 0-form yields the zero 1-form
    points, triangles = generate_mesh(3)
    edges, _ = oriented_edges(triangles)
    d0 = build_d0(points, edges)
    const_u = np.ones(len(points))
    du = d0 @ const_u
    assert np.allclose(du, 0)

def run_all_tests_derivatives():
    test_d0_entries()
    test_d1_entries()
    test_d1_d0_composition_zero()
    test_d0_constant_function_zero()
    print("All Derivative tests passed.")