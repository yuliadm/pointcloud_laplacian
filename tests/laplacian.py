import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import open3d as o3d
import argparse



# Build kNN weighted graph Laplacian on init point cloud

def build_knn_laplacian(P, k=20, sigma=None, symmetrize=True):
    '''
    compute the Laplacian least squares.
  
    - P: (n,3) source points
    - k: # neighbors in the kNN algorithm
    - sigma: optional sigma for computing Gaussian weights
    - symmetrize: optional force symmetrical W
    '''
    n = P.shape[0]
    nn = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(P)
    dists, inds = nn.kneighbors(P)

    # drop self neighbor at column 0
    dists = dists[:, 1:]
    inds = inds[:, 1:]

    # choose sigma if not provided: median neighbor distance
    if sigma is None:
        sigma = np.median(dists)
        sigma = max(sigma, 1e-8)

    # weights: Gaussian of distance
    weights = np.exp(-(dists**2) / (2.0 * sigma**2))

    # build sparse W
    rows = np.repeat(np.arange(n), k)
    cols = inds.reshape(-1)
    vals = weights.reshape(-1)

    W = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    if symmetrize:
        # make undirected by max(W, W^T)
        W = W.maximum(W.T)

    d = np.array(W.sum(axis=1)).reshape(-1)
    D = sp.diags(d, offsets=0, shape=(n, n), format="csr")
    L = D - W
    return L



# Recompute correspondence

def recompute_Tcorr(Pcur, T):
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(T)
    _, idx = nn.kneighbors(Pcur)
    return T[idx[:, 0]]


# Solve deformation frames: (L^T L + λ I) X_t = L^T δ(t) + λ Y(t)

def laplacian_morph_frames(
    P, T, k=20, lam=10.0, n_frames=30,
    refresh_every=5, sigma=None
):
    '''
    Morph P -> T using Laplacian least squares.
    Recomputes both Laplacian (graph) and correspondences every `refresh_every` frames.

    - P: (n,3) source points
    - T: (m,3) target points (can be different size)
    - L_init: optional initial Laplacian (ignored after first refresh)
    '''

    n = P.shape[0]
    I = sp.eye(n, format="csr")

    ts = np.linspace(0.0, 1.0, n_frames)
    frames = []

    Pcur = P.copy()

    # Placeholders for updates
    L_current = None
    solve_P = None
    deltaP = None
    deltaT = None
    Tcorr = None

    for i, t in enumerate(ts):
        if (i % refresh_every == 0) or (solve_P is None):
            # 1) Refresh correspondences using current geometry
            Tcorr = recompute_Tcorr(Pcur, T)  # (n,3)

            # 2) Refresh Laplacian (graph) on current geometry
            L_current = build_knn_laplacian(Pcur, k=k, sigma=sigma, symmetrize=True)

            # 3) Refresh Laplacian coords + solver factorization
            deltaP = L_current @ Pcur
            deltaT = L_current @ Tcorr

            A = (L_current.T @ L_current) + lam * I
            solve_P = sl.factorized(A.tocsc())

        # Interpolate *targets* for this t
        delta_t = (1.0 - t) * deltaP + t * deltaT
        Tt = (1.0 - t) * P + t * Tcorr

        rhs = (L_current.T @ delta_t) + lam * Tt

        Pnext = np.empty_like(Pcur)
        for dim in range(3):
            Pnext[:, dim] = solve_P(rhs[:, dim])

        frames.append(Pnext)
        Pcur = Pnext

    return ts, frames


# Step 6. Visualize

def plotly_slider(frames, title="Laplacian Morph", marker_size=2):
    '''
    Plot the deformed points with a Viridis colorscale (criterion: 
    distance from origin for each point in the current frame).
    
    frames: source frames
    '''

    # Stable camera bounds across all frames
    all_pts = np.vstack(frames)
    xmin, ymin, zmin = all_pts.min(axis=0)
    xmax, ymax, zmax = all_pts.max(axis=0)

    def dist_from_origin(X):
        return np.sqrt((X**2).sum(axis=1))

    # Initial frame 
    X0 = frames[0]
    c0 = dist_from_origin(X0)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X0[:, 0], y=X0[:, 1], z=X0[:, 2],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=c0,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="‖x‖")
                ),
                name="Deformed"
            )
        ]
    )

    # Animation frames (update positions)
    plotly_frames = []
    for i, Xt in enumerate(frames):
        ct = dist_from_origin(Xt)
        plotly_frames.append(
            go.Frame(
                data=[go.Scatter3d(
                    x=Xt[:, 0], y=Xt[:, 1], z=Xt[:, 2],
                    marker=dict(color=ct) 
                )],
                name=str(i)
            )
        )

    fig.frames = plotly_frames

    # Slider steps
    steps = [
        dict(
            method="animate",
            args=[[str(i)], dict(mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0))],
            label=str(i),
        )
        for i in range(len(frames))
    ]

    fig.update_layout(
        title=title,
        sliders=[dict(active=0, currentvalue={"prefix": "frame: "}, pad={"t": 30}, steps=steps)],
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=60, redraw=True),
                                      transition=dict(duration=0),
                                      fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode="immediate")]),
            ],
            x=0.0, y=1.12
        )],
        scene=dict(                       # set camera view 
        xaxis=dict(autorange='reversed'), 
        camera=dict(
            up=dict(x=0, y=1, z=0),      
            eye=dict(x=0, y=0, z=2),      
            center=dict(x=0, y=0, z=0)
        ),
        dragmode='orbit'                
    )
    )

    fig.show()



# End-to-end wrapper: normalize data, compute Laplacian, morph, visualize

def main(P, T, k=20, lam=10.0, n_frames=30):

    # Normalize initial and target point clouds
    mu = P.mean(axis=0, keepdims=True)
    P0 = P - mu
    scale = np.sqrt((P0**2).sum(axis=1)).mean()

    P = P0 / (scale + 1e-12)
    T = (T - mu) / (scale + 1e-12)
    
    # Compute deformation frames
    ts, frames = laplacian_morph_frames(P, T, k=k, lam=lam, n_frames=n_frames, refresh_every=5, sigma=None)
    
    # Visualize
    plotly_slider(frames, title=f"Laplacian morph (k={k}, λ={lam})")



def main(P_path, T_path, k=20, lam=10.0, n_frames=30):

    pcd_init = o3d.io.read_point_cloud(P_path)
    P = np.asarray(pcd_init.points, dtype=np.float64)

    pcd_tgt = o3d.io.read_point_cloud(T_path)
    T = np.asarray(pcd_tgt.points, dtype=np.float64)

    # Normalize initial and target point clouds
    mu = P.mean(axis=0, keepdims=True)
    P0 = P - mu
    scale = np.sqrt((P0**2).sum(axis=1)).mean()

    P = P0 / (scale + 1e-12)
    T = (T - mu) / (scale + 1e-12)
    
    # Compute deformation frames
    ts, frames = laplacian_morph_frames(P, T, k=k, lam=lam, n_frames=n_frames, refresh_every=5, sigma=None)
    
    # Visualize
    plotly_slider(frames, title=f"Laplacian morph (k={k}, λ={lam})")
    print(f"Processing matrices with k={k}, lam={lam}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Laplacian Morph")
    
    parser.add_argument("p_path", type=str, help="Path to matrix P (.ply file)")
    parser.add_argument("t_path", type=str, help="Path to matrix T (.ply file)")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--lam", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=30)

    args = parser.parse_args()

    # REMOVE np.load() from here. Just pass the strings (paths) to main.
    try:
        main(
            args.p_path, 
            args.t_path, 
            k=args.k, 
            lam=args.lam, 
            n_frames=args.frames
        )
    except Exception as e:
        print(f"Error during execution: {e}")
