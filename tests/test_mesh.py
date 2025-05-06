from mesh.generation import generate_mesh, generate_perturbed_mesh
from mesh.plotting import plot_mesh
from mesh.geometry import circumcenter

def test_generate_mesh_basic():
    points, triangles = generate_mesh(4)
    assert points.shape[1] == 2
    assert triangles.shape[1] == 3
    assert len(points) > 0

def test_plot_functions():
    points, triangles = generate_mesh(4)
    plot_mesh(points, triangles)

def point_in_triangle(cc, p0, p1, p2, tol=1e-8):
    """
    Check via barycentric coordinates whether cc lies inside triangle (p0,p1,p2).
    """
    v0 = p2 - p0
    v1 = p1 - p0
    v2 = cc - p0
    denom = v0[0]*v1[1] - v1[0]*v0[1]
    beta  = (v2[0]*v1[1] - v1[0]*v2[1]) / denom
    gamma = (v0[0]*v2[1] - v2[0]*v0[1]) / denom
    alpha = 1 - beta - gamma
    return (alpha >= -tol) and (beta >= -tol) and (gamma >= -tol)

def test_perturbed_mesh_well_centered(n):
    pts, tris = generate_perturbed_mesh(n, shift_factor=0.05)
    for tri in tris:
        p0, p1, p2 = pts[tri]
        cc = circumcenter([p0, p1, p2])
        assert cc is not None, "degenerate triangle encountered"
        assert point_in_triangle(cc, p0, p1, p2), f"triangle {tri} not well-centered"


def run_all_tests_mesh():
    test_generate_mesh_basic()
    test_plot_functions()
    for n in [4,8,16,32,64,128,256,512,1024,2048]:
        test_perturbed_mesh_well_centered(n)
    print("All mesh tests passed.")