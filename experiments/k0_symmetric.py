import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mesh.generation import generate_mesh, oriented_edges
from dec.derivatives import build_d0, build_d1
from dec.hodge import build_H0, build_H1
from experiments.manufacturing import u_ex, f_rhs

resolutions = [4,8,16,32,64,128,256,512]
errors_e_u = []
errors_de_u = []
hs = []

with open("k0_symmetric_convergence.txt", "w") as f_out:
    f_out.write(f"{'n':>4} {'h':>8} {'e_u error':>12} {'e_u rate':>10} {'de_u error':>12} {'de_u rate':>10}\n")
    f_out.write("-" * 108 + "\n")

    for i, n in enumerate(resolutions):
        points, triangles = generate_mesh(n)
        edges, edge_map = oriented_edges(triangles)
        d0 = build_d0(points, edges)
        d1 = build_d1(triangles, edge_map)
        H0 = build_H0(points, edges, triangles, edge_map)
        H1 = build_H1(points, edges, triangles, edge_map)
        H0_inv = sp.diags(1 / H0.diagonal())

        Pi_u = np.array([u_ex(x, y) for x, y in points])
        Pi_f = np.array([f_rhs(x, y) for x, y in points])
                        
        L = H0_inv @ d0.T @ H1 @ d0 #Note that (-1) from \delta and (-1) from (-d0.T) cancel

        u_h = spla.spsolve(L, Pi_f)

        e_u = Pi_u - u_h
        e_u -= (H0 @ e_u).sum() / H0.sum() #subtract average

        # |||e_u|||
        err_e_u = np.sqrt((e_u.T @ (H0 @ e_u)))
        errors_e_u.append(err_e_u)

        # |||d e_u|||
        de_u = d0 @ e_u
        err_de_u = np.sqrt((de_u.T @ (H1 @ de_u)))
        errors_de_u.append(err_de_u)
        
        h = 1 / n
        hs.append(h)
        

        if i == 0:
            f_out.write(f"{n:4d} {h:8.4f} {err_e_u:12.4e} {'-':>10} {err_de_u:12.4e} {'-':>10}\n")
        else:
            rate_L2   = np.log(errors_e_u[i-1] / err_e_u)  / np.log(hs[i-1]/h)
            rate_H1   = np.log(errors_de_u[i-1] / err_de_u)  / np.log(hs[i-1]/h)
            f_out.write(f"{n:4d} {h:8.4f} {err_e_u:12.4e} {rate_L2:10.2f} {err_de_u:12.4e} {rate_H1:10.2f}\n")