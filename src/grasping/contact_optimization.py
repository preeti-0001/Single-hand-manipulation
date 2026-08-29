from __future__ import annotations

import time
from pathlib import Path

import genesis as gs
import numpy as np
from src.utils.common import load_c2r, load_robot_qpos_on_video_timeline, resolve_episode 
from scipy.spatial.transform import Rotation



# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("hrdexdb")

HAND = "allegro_v5"
OBJECT_NAME = "apple"
SCENE = "4"

FPS = 30.0


# ============================================================
# CONTACT EXTRACTION
# ============================================================

def _to_numpy(x):
    """Convert a Genesis tensor/array to a NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)

# ============================================================
# CONTACT OPTIMIZATION
# ============================================================

def temporal_thresholding(validity_mask, min_duration=3, max_gap=1):
    """Remove short contacts and fill small temporal gaps."""
    mask = np.asarray(validity_mask, dtype=bool).copy()

    if mask.ndim != 3:
        raise ValueError(f"Expected (T,N,K), got {mask.shape}")

    T, N, K = mask.shape

    for n in range(N):
        for k in range(K):
            x = mask[:, n, k]

            # Fill gaps of <= max_gap frames.
            if max_gap > 0:
                for gap in range(1, max_gap + 1):
                    for t in range(gap, T - gap):
                        if not x[t] and x[t-gap] and x[t+gap]:
                            x[t] = True

            # Remove short runs.
            if min_duration > 1:
                start = None
                for t in range(T + 1):
                    active = t < T and x[t]

                    if active and start is None:
                        start = t
                    elif not active and start is not None:
                        end = t
                        if end - start < min_duration:
                            x[start:end] = False
                        start = None

            mask[:, n, k] = x

    return mask


def sparse_contact_selection(
    contact_tensor,
    validity_mask,
    sparsity_lambda=0.1,
    min_contacts=2,
    max_contacts=None,
):
    """
    Select a sparse subset of active contacts.

    Contacts closest to the frame-wise contact centroid are preferred.
    """
    C = np.asarray(contact_tensor, dtype=float)
    M = np.asarray(validity_mask, dtype=bool)

    if C.ndim != 4 or C.shape[-1] != 3:
        raise ValueError(f"Expected C=(T,N,K,3), got {C.shape}")
    if M.shape != C.shape[:3]:
        raise ValueError(f"Mask {M.shape} != {C.shape[:3]}")

    T, N, K, _ = C.shape
    out = np.zeros_like(M)

    for t in range(T):
        for n in range(N):
            ids = np.flatnonzero(M[t, n])
            if len(ids) == 0:
                continue

            if len(ids) <= min_contacts:
                out[t, n, ids] = True
                continue

            points = C[t, n, ids]
            centroid = points.mean(axis=0)
            cost = np.linalg.norm(points - centroid, axis=1)

            scale = np.mean(cost) + 1e-8
            score = cost / scale + sparsity_lambda

            order = ids[np.argsort(score)]
            count = len(order) if max_contacts is None else min(len(order), max_contacts)
            count = max(count, min_contacts)

            out[t, n, order[:count]] = True

    return out


def grasp_stability(
    contact_points,
    contact_normals=None,
    contact_weights=None,
    min_contacts=2,
):
    """
    Geometry-based grasp stability heuristic in [0,1].

    This is intentionally a lightweight heuristic; it is not a
    Ferrari-Canny force-closure calculation.
    """
    P = np.asarray(contact_points, dtype=float)

    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Expected points=(K,3), got {P.shape}")

    K = len(P)
    if K < min_contacts:
        return 0.0

    if contact_weights is None:
        w = np.ones(K)
    else:
        w = np.asarray(contact_weights, dtype=float)
        if w.shape != (K,):
            raise ValueError("contact_weights must have shape (K,)")

    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        return 0.0
    w /= w.sum()

    centroid = np.sum(P * w[:, None], axis=0)
    spread = np.sum(w * np.linalg.norm(P - centroid, axis=1))

    # 20 mm is used only as a normalization scale.
    spread_score = np.clip(spread / 0.02, 0.0, 1.0)
    count_score = np.clip(
        (K - min_contacts + 1) / 4.0,
        0.0,
        1.0,
    )

    score = 0.5 * count_score + 0.5 * spread_score

    if contact_normals is not None:
        normals = np.asarray(contact_normals, dtype=float)
        if normals.shape != P.shape:
            raise ValueError("Normals must have shape (K,3)")

        normals /= np.maximum(
            np.linalg.norm(normals, axis=1, keepdims=True),
            1e-8,
        )

        dots = np.clip(normals @ normals.T, -1.0, 1.0)
        pairwise = dots[np.triu_indices(K, k=1)]

        if pairwise.size:
            diversity = float(np.mean((1.0 - pairwise) / 2.0))
            score = 0.7 * score + 0.3 * diversity

    return float(np.clip(score, 0.0, 1.0))


def optimized_contact_mask(
    contact_tensor,
    validity_mask,
    min_duration=3,
    max_gap=1,
    sparsity_lambda=0.1,
    min_contacts=2,
    max_contacts=None,
    stability_threshold=0.25,
):
    """Run temporal filtering -> sparse selection -> stability filtering."""

    C = np.asarray(contact_tensor, dtype=float)
    M = np.asarray(validity_mask, dtype=bool)

    temporal_mask = temporal_thresholding(
        M,
        min_duration=min_duration,
        max_gap=max_gap,
    )

    sparse_mask = sparse_contact_selection(
        C,
        temporal_mask,
        sparsity_lambda=sparsity_lambda,
        min_contacts=min_contacts,
        max_contacts=max_contacts,
    )

    optimized = np.zeros_like(sparse_mask)

    T, N, K, _ = C.shape

    for t in range(T):
        for n in range(N):
            ids = np.flatnonzero(sparse_mask[t, n])
            if len(ids) == 0:
                continue

            score = grasp_stability(
                C[t, n, ids],
                min_contacts=min_contacts,
            )

            if score >= stability_threshold:
                optimized[t, n, ids] = True

    return optimized


def optimize_contacts(
    contact_tensor,
    validity_mask,
    min_duration=3,
    max_gap=1,
    sparsity_lambda=0.1,
    min_contacts=2,
    max_contacts=None,
    stability_threshold=0.25,
):
    """Return optimized contact tensor and optimized validity mask."""

    optimized_mask = optimized_contact_mask(
        contact_tensor,
        validity_mask,
        min_duration=min_duration,
        max_gap=max_gap,
        sparsity_lambda=sparsity_lambda,
        min_contacts=min_contacts,
        max_contacts=max_contacts,
        stability_threshold=stability_threshold,
    )

    optimized_tensor = np.where(
        optimized_mask[..., None],
        contact_tensor,
        0.0,
    )

    return optimized_tensor, optimized_mask


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # Resolve HRDexDB episode
    # ========================================================

    ep = resolve_episode(
        dataset_root=DATASET_ROOT,
        hand=HAND,
        object_name=OBJECT_NAME,
        scene=SCENE,
    )
    
    output_dir = (
        ep.episode_root
        / "processed"
    )
    

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    contact_file = output_dir / "contact_tensor.npy"
    mask_file = output_dir / "validity_mask.npy"

    contact_tensor = np.load(contact_file)
    validity_mask = np.load(mask_file)

    optimized_contact_tensor, optimized_validity_mask = optimize_contacts(
        contact_tensor=contact_tensor,
        validity_mask=validity_mask,
        min_duration=3,
        max_gap=1,
        sparsity_lambda=0.1,
        min_contacts=2,
        max_contacts=None,
        stability_threshold=0.25,
    )

    optimized_contact_file = (
        output_dir
        / "optimized_contact_tensor.npy"
    )

    optimized_mask_file = (
        output_dir
        / "optimized_validity_mask.npy"
    )



    np.save(
        optimized_contact_file,
        optimized_contact_tensor,
    )

    np.save(
        optimized_mask_file,
        optimized_validity_mask,
    )

    
    print("\n========== CONTACT RESULTS ==========")
    print("Contact tensor :", contact_tensor.shape)
    print("Validity mask  :", validity_mask.shape)
    print(
        "Total active link contacts :",
        int(validity_mask.sum()),
    )
    print("Optimized Contact tensor :", optimized_contact_tensor.shape)
    print("Optimized Validity mask  :", optimized_validity_mask.shape)
    print(
        " Total active link optimized contacts :",
        int(optimized_validity_mask.sum()),
    )
    print("Saved:", optimized_contact_file)
    print("Saved:", optimized_mask_file)
    print("=====================================\n")



if __name__ == "__main__":
    main()