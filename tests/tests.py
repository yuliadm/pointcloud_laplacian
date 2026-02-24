import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
from sklearn.neighbors import NearestNeighbors
import laplacian as lap
import argparse


def run_unit_tests(P=None, k=20, normalize=False):

    # 1. Laplacian tests

    # 1a. shape test
    n = P.shape[0]
    L, d = lap.build_knn_laplacian(P, k=k, sigma=None, symmetrize=True, normalize=normalize)
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
    
    if normalize:
        d_sqrt_values = np.sqrt(d)
        # create the diagonal matrix D^{1/2}
        D_sqrt = np.diag(d_sqrt_values)
        mat = D_sqrt @ np.eye(n)
        s = L @ mat

        is_within_tol = np.isclose(np.sum(s), 0, rtol=1e-4)

        if is_within_tol:
            print("Test 3 (Row-sum zero propery of normalized L) complete successfully")
        else:
            print("Test 3 (Row-sum zero propery of normalized L) failed")
    else:     
        row_sums = L.sum(axis=1).A1  # sum across rows, convert to a dense array (for a sparse mat)

        is_within_tol = np.isclose(np.sum(row_sums), 0, rtol=1e-4)

        if is_within_tol:
            print("Test 3 (Row-sum zero propery of unnormalized L) complete successfully")
        else:
            print("Test 3 (Row-sum zero propery of unnormalized L) failed")

    


    # 2. Differential coordinates tests: translationn invariance

    shift = np.array([1, 1, 1])

    translated_points = P + shift  # shift all points

    # Laplacian coordinates for original and translated points
    delta_original = L @ P
    delta_translated = L @ translated_points

    # Check if the **relative differences** in the Laplacian coordinates are the same
    delta_diff = np.linalg.norm(delta_original - delta_translated, axis=1)

    if normalize: 
        thres = 5e-1  # A small threshold for allowing the minor differences due to normalization
    else:
        thres = 1e-5 # exact for the unnormalized
    is_invariant = np.all(delta_diff < thres)

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



def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run unit tests on the Laplacian computation")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument('--normalize', action='store_true', help="Enable normalization (default is False)")
    
    # Parse arguments
    args = parser.parse_args()

    # Generate random point cloud P
    n_points = 1000

    # Generate random points in spherical coordinates
    phi = np.random.uniform(0, 2 * np.pi, n_points)  # Azimuthal angle (0 to 2π)
    theta = np.random.uniform(0, np.pi, n_points)    # Polar angle (0 to π)
    r = np.random.uniform(0, 1, n_points) ** (1/3)   # Radius (scaled for uniform distribution)

    # Convert spherical coordinates to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Stack them into a point cloud array
    P = np.column_stack((x, y, z))
    P = P.astype(np.float64)
    
    # Run unit tests with normalize flag
    run_unit_tests(P, k=20, normalize=args.normalize)


if __name__ == "__main__":
    main()