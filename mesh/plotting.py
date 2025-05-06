import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def plot_mesh(points, triangles, show_points=False, figsize=(6, 6), color='k'):
    plt.figure(figsize=figsize)
    tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    plt.triplot(tri, color=color, lw=0.8)
    if show_points:
        plt.plot(points[:, 0], points[:, 1], 'o', markersize=3)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()