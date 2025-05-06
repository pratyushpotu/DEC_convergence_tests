import numpy as np

def triangle_area(pts):
    A, B, C = pts
    return 0.5 * abs(np.cross(B - A, C - A))

def circumcenter(verts):
    A = verts[1] - verts[0]
    B = verts[2] - verts[0]
    A_perp = np.array([-A[1], A[0]])
    B_perp = np.array([-B[1], B[0]])
    midA = (verts[0] + verts[1]) / 2
    midB = (verts[0] + verts[2]) / 2
    A_matrix = np.stack([A_perp, -B_perp], axis=1)
    rhs = midB - midA
    t = np.linalg.solve(A_matrix, rhs)
    return midA + t[0] * A_perp