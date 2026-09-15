from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.utils.common import resolve_episode, load_robot_qpos_on_video_timeline


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("hrdexdb")
FPS = 30.0


# ============================================================
# GRASP CANDIDATE
# ============================================================

@dataclass
class GraspCandidate:
    position: torch.Tensor
    orientation: torch.Tensor | None
    qpos: torch.Tensor

    contact_links: torch.Tensor

    contact_score: float
    stability_score: float
    duration_score: float

    grip_type: str

    total_score: float = 0.0


# ============================================================
# CONTACT EVENTS
# ============================================================

def extract_contact_events(
    validity_mask,
    min_duration=3,
):
    """
    Extract continuous contact intervals.

    validity_mask:
        [T, N, K]

    Returns:
        List of (start_frame, end_frame)
    """

    mask = np.asarray(validity_mask, dtype=bool)

    if mask.ndim != 3:
        raise ValueError(
            f"Expected validity mask [T,N,K], got {mask.shape}"
        )

    # Contact exists if ANY hand/link has contact
    active = mask.any(axis=(1, 2))

    events = []

    start = None

    for t, is_active in enumerate(active):

        if is_active and start is None:
            start = t

        elif not is_active and start is not None:

            end = t - 1

            if end - start + 1 >= min_duration:
                events.append((start, end))

            start = None

    # Handle contact continuing until final frame
    if start is not None:

        end = len(active) - 1

        if end - start + 1 >= min_duration:
            events.append((start, end))

    return events


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def make_grasp_candidate(
    start,
    end,
    contact_tensor,
    validity_mask,
    robot_qpos,
):
    """
    Create one grasp candidate from one contact event.

    contact_tensor:
        [T,N,K,3]

    validity_mask:
        [T,N,K]

    robot_qpos:
        [T,D]
    """

    C = np.asarray(contact_tensor)
    M = np.asarray(validity_mask)

    contacts = C[start:end + 1]
    valid = M[start:end + 1]

    # --------------------------------------------------------
    # Valid contact points
    # --------------------------------------------------------

    valid_points = contacts[valid]

    if len(valid_points) == 0:
        raise ValueError(
            f"No valid contacts found between {start} and {end}"
        )

    # --------------------------------------------------------
    # Grasp position
    # --------------------------------------------------------

    position_np = valid_points.mean(axis=0)

    position = torch.tensor(
        position_np,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Representative robot configuration
    # --------------------------------------------------------

    mid = (start + end) // 2

    qpos = torch.tensor(
        robot_qpos[mid],
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Contact links
    # --------------------------------------------------------

    contact_links_np = valid.any(axis=(0, 1))

    contact_links = torch.tensor(
        contact_links_np,
        dtype=torch.bool,
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    contact_count = int(contact_links_np.sum())

    duration = end - start + 1

    contact_score = float(contact_count)

    duration_score = float(duration)

    # Stability will be calculated separately
    stability_score = 0.0

    return GraspCandidate(
        position=position,
        orientation=None,
        qpos=qpos,
        contact_links=contact_links,
        contact_score=contact_score,
        stability_score=stability_score,
        duration_score=duration_score,
        grip_type="unknown",
    )


# ============================================================
# STABILITY
# ============================================================

def compute_stability(
    candidate,
    contact_tensor,
    validity_mask,
    start,
    end,
):
    """
    Compute a simple geometric stability score [0,1].

    This is a heuristic and NOT a Ferrari-Canny
    force-closure metric.
    """

    C = np.asarray(contact_tensor)
    M = np.asarray(validity_mask)

    contacts = C[start:end + 1]
    valid = M[start:end + 1]

    points = contacts[valid]

    if len(points) < 2:
        return 0.0

    centroid = points.mean(axis=0)

    distances = np.linalg.norm(
        points - centroid,
        axis=1,
    )

    spread = distances.mean()

    # Normalize approximately around 20 mm
    spread_score = np.clip(
        spread / 0.02,
        0.0,
        1.0,
    )

    count = len(points)

    count_score = np.clip(
        count / 10.0,
        0.0,
        1.0,
    )

    score = (
        0.5 * spread_score
        + 0.5 * count_score
    )

    return float(np.clip(score, 0.0, 1.0))


# ============================================================
# GRASP SCORING
# ============================================================

def score_grasp(candidate):

    """
    Combine the three grasp properties.

    NOTE:
    These weights are initial values.
    Tune them experimentally later.
    """

    candidate.total_score = (
        0.4 * candidate.contact_score
        + 0.3 * candidate.duration_score
        + 0.3 * candidate.stability_score
    )

    return candidate


# ============================================================
# SELECT IDEAL GRASPS
# ============================================================

def select_ideal_grasps(
    candidates,
    k=5,
):

    candidates = sorted(
        candidates,
        key=lambda g: g.total_score,
        reverse=True,
    )

    return candidates[:k]


# ============================================================
# SAVE CANDIDATES
# ============================================================

def save_candidates(
    candidates,
    output_file,
):

    data = []

    for i, grasp in enumerate(candidates):

        data.append(
            {
                "id": i,
                "position": grasp.position.numpy(),
                "qpos": grasp.qpos.numpy(),
                "contact_links": grasp.contact_links.numpy(),
                "contact_score": grasp.contact_score,
                "stability_score": grasp.stability_score,
                "duration_score": grasp.duration_score,
                "grip_type": grasp.grip_type,
                "total_score": grasp.total_score,
            }
        )

    np.save(
        output_file,
        np.array(data, dtype=object),
        allow_pickle=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate grasp candidates from optimized contacts."
    )

    parser.add_argument(
        "--hand",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--object_name",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--scene",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Resolve episode
    # --------------------------------------------------------

    ep = resolve_episode(
        dataset_root=DATASET_ROOT,
        hand=args.hand,
        object_name=args.object_name,
        scene=args.scene,
    )

    output_dir = ep.episode_root / "processed"

    # --------------------------------------------------------
    # Load optimized contacts
    # --------------------------------------------------------

    contact_file = (
        output_dir /
        "optimized_contact_tensor.npy"
    )

    mask_file = (
        output_dir /
        "optimized_validity_mask.npy"
    )

    contact_tensor = np.load(contact_file)

    validity_mask = np.load(mask_file)

    # --------------------------------------------------------
    # Load robot qpos
    # --------------------------------------------------------

    # CHANGE THIS PATH TO YOUR ACTUAL QPOS FILE
    
    qpos, video_time, frame_ids, hand_dof, arm_dof = load_robot_qpos_on_video_timeline(
        ep.episode_root,
        ep.hand,
    )

    # --------------------------------------------------------
    # Extract contact events
    # --------------------------------------------------------

    events = extract_contact_events(
        validity_mask,
        min_duration=3,
    )

    print("\n========== GRASP CANDIDATES ==========")

    print(
        "Optimized contact tensor:",
        contact_tensor.shape,
    )

    print(
        "Optimized validity mask:",
        validity_mask.shape,
    )

    print(
        "Contact events:",
        len(events),
    )

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    candidates = []

    for start, end in events:

        candidate = make_grasp_candidate(
            start=start,
            end=end,
            contact_tensor=contact_tensor,
            validity_mask=validity_mask,
            robot_qpos=qpos,
        )

        candidate.stability_score = compute_stability(
            candidate,
            contact_tensor,
            validity_mask,
            start,
            end,
        )

        score_grasp(candidate)

        candidates.append(candidate)

    # --------------------------------------------------------
    # Select top K
    # --------------------------------------------------------

    ideal_grasps = select_ideal_grasps(
        candidates,
        k=args.top_k,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    for i, grasp in enumerate(ideal_grasps):

        print(
            f"\nGrasp {i}"
        )

        print(
            "Position:",
            grasp.position.numpy(),
        )

        print(
            "Contact links:",
            grasp.contact_links.numpy(),
        )

        print(
            "Contact score:",
            grasp.contact_score,
        )

        print(
            "Duration score:",
            grasp.duration_score,
        )

        print(
            "Stability score:",
            grasp.stability_score,
        )

        print(
            "Total score:",
            grasp.total_score,
        )

        print(
            "Grip type:",
            grasp.grip_type,
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        output_dir /
        "grasp_candidates.npy"
    )

    save_candidates(
        candidates,
        output_file,
    )

    ideal_file = (
        output_dir /
        "ideal_grasps.npy"
    )

    save_candidates(
        ideal_grasps,
        ideal_file,
    )

    print("\nSaved:")
    print(output_file)
    print(ideal_file)

    print("======================================\n")


if __name__ == "__main__":
    main()