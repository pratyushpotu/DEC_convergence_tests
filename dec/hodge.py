import numpy as np
import scipy.sparse as sp
from mesh.geometry import triangle_area, circumcenter

def build_H0(points, edges, triangles, edge_map):
    n_vertices = len(points)
    n_edges = len(edges)

    # Compute circumcenters of triangles
    face_cc = np.array([circumcenter(points[tri]) for tri in triangles])

    # For each vertex, collect its dual cell area
    vertex_area = np.zeros(n_vertices)

    # Build edge-to-face incidence
    edge_faces = [[] for _ in range(n_edges)]
    for t_idx, tri in enumerate(edge_map):
        for e_idx, _ in tri:
            edge_faces[e_idx].append(t_idx)

    # Build vertex-to-face map
    vertex_faces = [[] for _ in range(n_vertices)]
    for t_idx, tri in enumerate(triangles):
        for v in tri:
            vertex_faces[v].append(t_idx)

    # Compute *0 via circumcentric dual cell
    for v_idx in range(n_vertices):
        adjacent_faces = vertex_faces[v_idx]
        p0 = points[v_idx]

        for f_idx in adjacent_faces:
            tri = triangles[f_idx]
            cc = face_cc[f_idx]

            # Find the two edges in triangle that touch v_idx
            v_indices = list(tri)
            i = v_indices.index(v_idx)
            v_prev = v_indices[(i + 2) % 3]
            v_next = v_indices[(i + 1) % 3]

            mid1 = (points[v_idx] + points[v_prev]) / 2
            mid2 = (points[v_idx] + points[v_next]) / 2

            # Polygon with points: v_idx, mid1, cc, mid2
            quad = np.array([p0, mid1, cc, mid2])

            # Split quad into two triangles and sum their areas
            area = 0.5 * abs(np.cross(quad[1] - quad[0], quad[2] - quad[0]))
            area += 0.5 * abs(np.cross(quad[3] - quad[0], quad[2] - quad[0]))
            vertex_area[v_idx] += area

    H0 = sp.diags(vertex_area)
    return H0

def build_H1(points, edges, triangles, edge_map):
    # *1: dual edge length / primal edge length
    n_edges = len(edges)
    edge_lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    dual_lengths = np.zeros(n_edges)

    # Compute circumcenters of triangles
    face_cc = np.array([circumcenter(points[tri]) for tri in triangles])

    # Build edge-to-face incidence
    edge_faces = [[] for _ in range(n_edges)]
    for t_idx, tri in enumerate(edge_map):
        for e_idx, _ in tri:
            edge_faces[e_idx].append(t_idx)

    for i, face_list in enumerate(edge_faces):
        if len(face_list) == 2:
            c1, c2 = face_cc[face_list[0]], face_cc[face_list[1]]
        elif len(face_list) == 1:
            cc = face_cc[face_list[0]]
            mid = (points[edges[i][0]] + points[edges[i][1]]) / 2
            c1, c2 = cc, mid
        else:
            raise RuntimeError("Edge belongs to more than 2 faces?")
        dual_lengths[i] = np.linalg.norm(c2 - c1)

    H1 = sp.diags(dual_lengths / edge_lengths)
    return H1

def build_H2(points, triangles):
    # *2: 1 / primal triangle area
    tri_areas = np.array([triangle_area(points[tri]) for tri in triangles])
    H2 = sp.diags(np.reciprocal(tri_areas))
    return H2