import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from mesh.generation import generate_perturbed_mesh, oriented_edges
from dec.derivatives import build_d0, build_d1
from dec.hodge import build_H0, build_H1, build_H2
from dec.integrators import compute_1_cochain
from experiments.manufacturing import u_vector, f_vector, delta_u

resolutions = [4,8,16,32,64,128,256,512,1024,2048]

hs = []
errors_e_u = []
errors_de_u = []
errors_e_rho  = []
errors_de_rho = []

with open("k1_perturbed_convergence.txt", "w") as f_out:
    # Header: n, h, e_u, rate_e_u, de_u, rate_de_u, e_rho, rate_e_rho
    f_out.write(f"{'n':>4} {'h':>8} {'e_u error':>12} {'e_u rate':>10} "
                f"{'de_u error':>12} {'de_u rate':>10} "
                f"{'e_rho error':>12} {'e_rho rate':>12} "
                f"{'de_rho_err':>15} {'rate_de_rho':>14}\n"
            )
    f_out.write("-" * 132 + "\n")

    for i, n in enumerate(resolutions):
        points, triangles = generate_perturbed_mesh(n)
        edges, edge_map   = oriented_edges(triangles)
        d0 = build_d0(points, edges)
        d1 = build_d1(triangles, edge_map)
        H0 = build_H0(points, edges, triangles, edge_map)
        H1 = build_H1(points, edges, triangles, edge_map)
        H2 = build_H2(points, triangles)

        H0_inv = sp.diags(1.0 / H0.diagonal())
        H1_inv = sp.diags(1.0 / H1.diagonal())

        # Assemble k=1 Hodge-Laplacian
        L = d0 @ H0_inv @ d0.T @ H1 + H1_inv @ d1.T @ H2 @ d1 #Note that (-1) and (-1) cancel in first term

        Pi_u = compute_1_cochain(points, edges, u_vector)
        Pi_f = compute_1_cochain(points, edges, f_vector)

        # Solve
        u_h = spla.spsolve(L, Pi_f)

        #|||e_u|||
        e_u = Pi_u - u_h
        err_e_u = np.sqrt((e_u.T @ (H1 @ e_u)))
        errors_e_u.append(err_e_u)

        #|||de_u|||
        d_e_u   = d1 @ e_u
        err_de_u = np.sqrt((d_e_u.T @ (H2 @ d_e_u)))
        errors_de_u.append(err_de_u)

        #|||e_rho|||
        Pi_rho  = np.array([delta_u(x, y) for x, y in points])
        rho_h   = H0_inv @ d0.T @ H1 @ u_h

        e_rho   = Pi_rho - rho_h
        err_e_rho = np.sqrt((e_rho.T @ (H0 @ e_rho)))
        errors_e_rho.append(err_e_rho)

        #|||de_rho|||
        de_rho = d0 @ e_rho
        err_de_rho = np.sqrt(de_rho.T @ (H1 @ de_rho))
        errors_de_rho.append(err_de_rho)

        h = 1.0 / n
        hs.append(h)

        if i == 0:
            f_out.write(f"{n:4d} {h:8.4f} "
                        f"{err_e_u:12.4e} {'-':>10} "
                        f"{err_de_u:12.4e} {'-':>10} "
                        f"{err_e_rho:12.4e} {'-':>12} "
                        f"{err_de_rho:15.4e} {'-':>14}\n"
            )
        else:
            rate_e_u     = np.log(errors_e_u[i-1] / err_e_u) / np.log(hs[i-1]/h)
            rate_de_u    = np.log(errors_de_u[i-1] / err_de_u) / np.log(hs[i-1]/h)
            rate_e_rho   = np.log(errors_e_rho[i-1] / err_e_rho) / np.log(hs[i-1]/h)
            rate_de_rho  = np.log(errors_de_rho[i-1] / err_de_rho) / np.log(hs[i-1]/h)
            f_out.write(f"{n:4d} {h:8.4f} "
                        f"{err_e_u:12.4e} {rate_e_u:10.2f} "
                        f"{err_de_u:12.4e} {rate_de_u:10.2f} "
                        f"{err_e_rho:12.4e} {rate_e_rho:12.2f} "
                        f"{err_de_rho:15.4e} {rate_de_rho:14.2f}\n"
                    )
