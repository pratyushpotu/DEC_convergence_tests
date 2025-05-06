import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mesh.generation import generate_perturbed_mesh, oriented_edges
from dec.derivatives import build_d1
from dec.hodge import build_H1, build_H2
from dec.integrators import compute_2_cochain, compute_1_cochain
from experiments.manufacturing import u_ex, f_rhs, delta2_u

resolutions = [4,8,16,32,64,128,256,512,1024,2048]

hs = []
errors_e_u = []
errors_e_rho = []

with open("k2_perturbed_convergence.txt", "w") as f_out:
    # Header: n, h, e_u error, e_u rate, e_rho error, e_rho rate
    f_out.write(
        f"{'n':>4} {'h':>8} {'e_u error':>12} {'e_u rate':>10} "
        f"{'e_rho error':>15} {'e_rho rate':>12}\n"
    )
    f_out.write("-" * 66 + "\n")

    for i, n in enumerate(resolutions):
        points, triangles = generate_perturbed_mesh(n)
        edges, edge_map   = oriented_edges(triangles)
        d1 = build_d1(triangles, edge_map)
        H1 = build_H1(points, edges, triangles, edge_map)
        H2 = build_H2(points, triangles)

        H1_inv = sp.diags(1.0 / H1.diagonal())

        # Assemble k=2 Hodge-Laplacian
        L = d1 @ H1_inv @ d1.T @ H2

        Pi_u = compute_2_cochain(points, triangles, u_ex)
        Pi_f = compute_2_cochain(points, triangles, f_rhs)

        u_h = spla.spsolve(L, Pi_f)

        # |||e_u|||
        e_u = Pi_u - u_h
        err_e_u = np.sqrt((e_u.T @ (H2 @ e_u)))
        errors_e_u.append(err_e_u)

        # |||e_rho|||
        Pi_rho = compute_1_cochain(points, edges, delta2_u)
        rho_h = H1_inv @ d1.T @ H2 @ u_h

        e_rho = Pi_rho - rho_h
        err_rho = np.sqrt((e_rho.T @ (H1 @ e_rho)))
        errors_e_rho.append(err_rho)

        h = 1.0 / n
        hs.append(h)

        if i == 0:
            f_out.write(
                f"{n:4d} {h:8.4f} "
                f"{err_e_u:12.4e} {'-':>10} "
                f"{err_rho:15.4e} {'-':>12}\n"
            )
        else:
            rate_e = np.log(errors_e_u[i-1] / err_e_u) / np.log(hs[i-1]/h)
            rate_rho  = np.log(errors_e_rho[i-1] / err_rho) / np.log(hs[i-1]/h)
            f_out.write(
                f"{n:4d} {h:8.4f} "
                f"{err_e_u:12.4e} {rate_e:10.2f} "
                f"{err_rho:15.4e} {rate_rho:12.2f}\n"
            )
