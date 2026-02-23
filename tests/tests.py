import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
from sklearn.neighbors import NearestNeighbors
import laplacian as lap



def run_unit_tests(P, k=20):

    # 1. Laplacian tests

    # 1a. shape test
    n = P.shape[0]
    L = lap.build_knn_laplacian(P, k=k, sigma=None, symmetrize=True)
    if L.shape == (n, n):
        print("Test 1 (Laplacian shape) complete successfully")
    else: 
        print("Test 1 (Laplacian shape) failed")
    
    #  1b. number of nonzero elements test
    total_nonzero = L.nnz
    is_within_tol = np.isclose(total_nonzero, n*k, rtol=0.35)
    if is_within_tol == True:
        print("Test 2 (# nonzero elements in L) complete successfully")
    else:
        print("Test 2 (# nonzero elements in L) failed")

    
    # 1c. row-sum zero propery test *(for the unnormalized L)
    row_sums = L.sum(axis=1).A1  # sum across rows, convert to a dense array (for a sparse mat)

    is_within_tol = np.isclose(np.sum(row_sums), 0, rtol=1e-4)

    if is_within_tol:
        print("Test 3 (Row-sum zero propery of L) complete successfully")
    else:
        print("Test 3 (Row-sum zero propery of L) failed")

    # 2. Differential coordinates tests: translationn invariance

    shift = np.array([1, 1, 1])

    translated_points = P + shift  # shift all points

    # Laplacian coordinates for original and translated points
    delta_original = L @ P
    delta_translated = L @ translated_points

    # Check if the **relative differences** in the Laplacian coordinates are the same
    is_invariant = np.allclose(delta_original, delta_translated, rtol=1e-5)

    if is_invariant:
        print("Test 4 (Translation invariance for differential coords) complete successfully")
    else:
        print("Test 4 (Translation invariance for differential coords) failed")


    # 3. End-to-end behavioral tests (small synthetic clouds)

    # 3a. “no-op” morph: P → P
    mu = P.mean(axis=0, keepdims=True)
    P0 = P - mu
    scale = np.sqrt((P0**2).sum(axis=1)).mean()

    P = P0 / (scale + 1e-12)

    # Compute deformation frames
    _, frames = lap.laplacian_morph_frames(P, P) 

    # Check if the deformation moves towards P monotonically (no-op)
    first_frame = frames[0]
    last_frame = frames[-1]

    # Check if the first frame and last frame are close (no-op means no change)
    if np.allclose(first_frame, last_frame, atol=1e-5):
        print("Test 5 (No-op morph) complete successfully")
    else:
        print("Test 5 (No-op morph) failed")



    # 3b. known deformation test: Deforming P to T (scaled along Z-axis)

    T = P.copy()
    T[:, 2] = 2 * P[:, 2]  # e.g. scaling Z by a factor of 2

    # Normalize
    T = (T - mu) / (scale + 1e-12)

    # Compute the deformation frames
    _, frames_ = lap.laplacian_morph_frames(P, T)

    # Check that the deformation moves towards T monotonically
    distances_to_T = [np.linalg.norm(frame - T, axis=1) for frame in frames_]
    distances_to_T = np.array(distances_to_T)

    # Calculate the mean distance to target T for each frame
    mean_distances_to_T = distances_to_T.mean(axis=1)

    # Check if the mean distances are monotonically decreasing (allow for tiny flustuations)
    is_monotonic = np.all(np.diff(mean_distances_to_T) <= 1e-2).item()  
    if is_monotonic: 
        print("Test 6 (Known Deformation) complete successfully")
    else:
        print("Test 6 (Known Deformation) failed")


if __name__ == "__main__":
    # generate a point cloud
    P = np.random.randn(1000, 3).astype(np.float64)
    run_unit_tests(P, k=20)