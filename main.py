#import os; os.environ["NUMBA_ENABLE_CUDASIM"] = "1"; os.environ["NUMBA_CUDA_DEBUGINFO"] = "1";
from numba import cuda, njit
import open3d as o3d
import numpy as np
from math import atan2, sqrt, pow, sin, cos

DIM = 500
N = 8
MAX_ITER = 20


@cuda.jit
def create_mandlebulb(grid2d):
    i, j, k = cuda.grid(3)

    x = 2 * i / DIM - 1
    y = 2 * j / DIM - 1
    z = 2 * k / DIM - 1

    zetaR = 0
    zetaT = 0
    zetaP = 0
    reachedEnd = True
    for iter in range(MAX_ITER):
        r = sqrt(zetaR * zetaR + zetaT * zetaT + zetaP * zetaP)

        if r > 2:
            reachedEnd = False
            break

        theta = atan2(sqrt(zetaR * zetaR + zetaT * zetaT), zetaP)
        phi = atan2(zetaT, zetaR)
        deltax = pow(r, N) * sin(theta * N) * cos(phi * N)
        deltay = pow(r, N) * sin(theta * N) * sin(phi * N)
        deltaz = pow(r, N) * cos(theta * N)

        zetaR = x + deltax
        zetaT = y + deltay
        zetaP = z + deltaz
    grid2d[i, j, k] = 0 if reachedEnd else 1


@njit
def mandlebulb_edge_coordinates(mandlebulb):
    mandlebulb_coords = []
    for xi, x in enumerate(mandlebulb):
        for yi, y in enumerate(x):
            last_z = 1
            for zi, z in enumerate(y):
                if z != last_z:
                    mandlebulb_coords.append([2 * xi / DIM - 1, 2 * yi / DIM - 1, 2 * zi / DIM - 1])
                last_z = z
    print("Number of edge coordinates found:", len(mandlebulb_coords))
    return mandlebulb_coords


if __name__ == '__main__':
    mandlebulb = np.zeros((DIM, DIM, DIM))
    
    print("Doing mandlebulb gpu calculation...")
    create_mandlebulb[(int(DIM/10),int(DIM//10),int(DIM//10)),(10,10,10)](mandlebulb)
    
    print("Finding edge coordinates...")
    mandlebulb_edge_cooridnates = np.array(mandlebulb_edge_coordinates(mandlebulb))
    
    print("Drawing pointcloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mandlebulb_edge_cooridnates)
    o3d.visualization.draw_geometries([pcd])