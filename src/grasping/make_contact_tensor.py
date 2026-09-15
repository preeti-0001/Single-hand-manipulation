from __future__ import annotations

import time
from pathlib import Path

import genesis as gs
import numpy as np
from src.utils.common import (
    load_object_trajectory,
    load_robot_qpos_on_video_timeline,
    resolve_episode,
)

from src.utils.math_utils import matrix_to_wxyz, _to_numpy
import argparse

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("hrdexdb")
FPS = 30.0


def compute_contacts(
    robot,
    obj,
    contact_tensor,
    validity_mask,
    frame,
    object_index=0,
):
    """
    Convert Genesis hand-object contacts into a DexMachina-style
    contact representation for one timestep.

    For a single rigid object (apple):

        contact_tensor : (T, 1, K, 3)
        validity_mask  : (T, 1, K)

    where K = number of robot links.

    Genesis gives contact positions in WORLD coordinates.  We keep
    them in world coordinates here because the replay itself is
    expressed in the robot/world frame.

    A link is considered valid when Genesis reports at least one
    hand-object contact for that link.  If multiple contacts belong
    to the same link, their contact positions are averaged.

    IMPORTANT:
    This uses Genesis' physics contacts, not the original
    DexMachina/MANO 10-mm KD-tree preprocessing. It is the practical
    contact extraction for your HRDexDB -> Allegro setup.
    """

    contacts = robot.get_contacts(with_entity=obj)

    if not contacts:
        return

    # Genesis returns a dictionary of parallel arrays.
    # For a non-parallel scene these are:
    #
    #   link_a/link_b : (n_contacts,)
    #   position     : (n_contacts, 3)
    #
    link_a = _to_numpy(contacts.get("link_a", []))
    link_b = _to_numpy(contacts.get("link_b", []))
    positions = _to_numpy(contacts.get("position", []))

    if positions.size == 0:
        return

    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    link_a = np.asarray(link_a).reshape(-1)
    link_b = np.asarray(link_b).reshape(-1)

    # For a single-environment Genesis scene, there is normally no
    # valid_mask field. Keep this robust in case one is returned.
    genesis_valid = contacts.get("valid_mask", None)
    if genesis_valid is not None:
        genesis_valid = _to_numpy(genesis_valid).astype(bool).reshape(-1)
    else:
        genesis_valid = np.ones(len(positions), dtype=bool)

    # Genesis stores global link indices. The robot entity has a
    # contiguous global link range.
    robot_link_start = int(robot.link_start)
    robot_link_end = robot_link_start + int(robot.n_links)

    # A contact pair contains one robot link and one object link.
    # Select whichever side belongs to the robot.
    robot_links = np.full(len(positions), -1, dtype=np.int64)

    a_is_robot = (link_a >= robot_link_start) & (link_a < robot_link_end)

    b_is_robot = (link_b >= robot_link_start) & (link_b < robot_link_end)

    robot_links[a_is_robot] = link_a[a_is_robot]
    robot_links[~a_is_robot & b_is_robot] = link_b[~a_is_robot & b_is_robot]

    valid_contacts = (
        genesis_valid
        & (robot_links >= robot_link_start)
        & (robot_links < robot_link_end)
    )

    if not np.any(valid_contacts):
        return

    # Convert global Genesis link index -> local robot link index.
    local_links = (robot_links[valid_contacts] - robot_link_start).astype(np.int64)

    contact_positions = positions[valid_contacts]

    # One representative contact point per robot link.
    #
    # This is analogous to DexMachina's reduction of multiple raw
    # contacts belonging to one hand link into one contact location.
    for local_link in np.unique(local_links):

        mask = local_links == local_link
        link_positions = contact_positions[mask]

        contact_tensor[
            frame,
            object_index,
            local_link,
        ] = np.mean(
            link_positions,
            axis=0,
        )

        validity_mask[
            frame,
            object_index,
            local_link,
        ] = True

def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay an HRDexDB episode in Genesis."
    )

    parser.add_argument(
        "--hand",
        required=True,
        type=str,
        help="Hand name, e.g. allegro_v5",
    )

    parser.add_argument(
        "--object_name",
        required=True,
        type=str,
        help="Object name, e.g. apple",
    )

    parser.add_argument(
        "--scene",
        required=True,
        type=str,
        help="Scene ID, e.g. 4",
    )

    return parser.parse_args()

def main():

    args = parse_args()

    HAND = args.hand
    OBJECT_NAME = args.object_name
    SCENE = args.scene

    # Resolve HRDexDB episode
    ep = resolve_episode(
        dataset_root=DATASET_ROOT,
        hand=HAND,
        object_name=OBJECT_NAME,
        scene=SCENE,
    )

    print("\n========== HRDexDB EPISODE ==========")
    print("Episode root :", ep.episode_root)
    print("Hand         :", ep.hand)
    print("Object       :", ep.object_name)
    print("Robot URDF   :", ep.robot_urdf)
    print("Object mesh  :", ep.object_mesh)
    print("=====================================\n")

    # Robot trajectory

    qpos, video_time, frame_ids, hand_dof, arm_dof = load_robot_qpos_on_video_timeline(
        ep.episode_root,
        ep.hand,
    )

    print("Robot qpos shape :", qpos.shape)
    print("Video time shape :", video_time.shape)
    print("Frame IDs shape  :", frame_ids.shape)

    # Object trajectory

    object_poses = load_object_trajectory(
        DATASET_ROOT,
        HAND,
        OBJECT_NAME,
        SCENE,
    )

    # Determine replay length

    robot_frames = len(qpos)
    object_frames = len(object_poses)

    timeline_len = min(
        robot_frames,
        object_frames,
    )

    print("\n========== REPLAY ==========")
    print("Robot frames  :", robot_frames)
    print("Object frames :", object_frames)
    print("Replay frames :", timeline_len)
    print("============================\n")

    # Genesis

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(
            dt=1.0 / FPS,
            gravity=(0.0, 0.0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.7, 0.7, 0.45),
            camera_lookat=(0.0, 0.0, 0.15),
            camera_fov=40,
            res=(1280, 720),
        ),
    )

    #  Ground

    plane = scene.add_entity(gs.morphs.Plane())

    # Robot

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(ep.robot_urdf),
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            # Required for your current URDF.
            recompute_inertia=True,
        )
    )

    # Object

    first_pose = object_poses[0]

    first_position = first_pose[:3, 3]

    first_quat = matrix_to_wxyz(first_pose[:3, :3])

    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=str(ep.object_mesh),
            pos=first_position,
            quat=first_quat,
            scale=1.0,
            # Kinematic replay false -> object will follow trajectory via given qpos/vel, not physics.
            fixed=False,
        ),
        material=gs.materials.Rigid(
            friction=1.5,
        ),
    )

    # Build

    scene.build()

    # Robot DOF check

    print("\n========== ROBOT ==========")
    print("Genesis qpos size :", robot.n_qs)
    print("Genesis DOFs      :", robot.n_dofs)
    print("HRDexDB qpos size :", qpos.shape[1])
    print("===========================\n")

    if robot.n_qs != qpos.shape[1]:

        raise RuntimeError(
            "\nDOF mismatch!\n"
            f"Genesis robot qpos : {robot.n_qs}\n"
            f"HRDexDB trajectory : {qpos.shape[1]}"
        )

    # Print joints

    print("Genesis joints:")

    for i, joint in enumerate(robot.joints):

        print(f"{i:02d} | {joint.name}")

    # ========================================================
    # Contact tensors
    #
    # One rigid object (apple), so:
    #
    #   contact_tensor : (T, 1, K, 3)
    #   validity_mask  : (T, 1, K)
    # T is frames
    # K is the number of links in the Genesis model.
    # ========================================================

    num_links = int(robot.n_links)

    contact_tensor = np.zeros(
        (timeline_len, 1, num_links, 3),
        dtype=np.float32,
    )

    validity_mask = np.zeros(
        (timeline_len, 1, num_links),
        dtype=bool,
    )

    print("\n========== CONTACT TENSOR ==========")
    print("Shape :", contact_tensor.shape)
    print("Mask  :", validity_mask.shape)
    print("Links :", num_links)
    print("====================================\n")

    # Replay

    print(f"\nStarting replay " f"({timeline_len} frames @ {FPS} FPS)\n")

    frame_dt = 1.0 / FPS

    for frame in range(timeline_len):
        start_time = time.perf_counter()
        # ROBOT
        robot.set_qpos(
            qpos[frame],
            zero_velocity=True,
        )

        # OBJECT: This is directly based on your working object-only Genesis code.
        T = object_poses[frame]

        # Position
        position = T[:3, 3]

        # Rotation
        rotation_matrix = T[:3, :3]
        quat_wxyz = matrix_to_wxyz(rotation_matrix)

        # Move object
        obj.set_pos(
            position,
            zero_velocity=True,
        )

        obj.set_quat(
            quat_wxyz,
            zero_velocity=True,
        )

        # Advance Genesis
        scene.step()

        # ====================================================
        # CONTACTS
        #
        # Convert Genesis hand-object contacts into:
        #
        #   contact_tensor[frame, 0, link, :] = contact xyz
        #   validity_mask[frame, 0, link]      = True/False
        # ====================================================
        compute_contacts(
            robot=robot,
            obj=obj,
            contact_tensor=contact_tensor,
            validity_mask=validity_mask,
            frame=frame,
        )

        # Debug: print only frames where a contact exists.
        if np.any(validity_mask[frame]):
            active_links = np.where(validity_mask[frame, 0])[0]

            print(
                f"Frame {frame:04d} | "
                f"contact links = {active_links.tolist()} | "
                f"contacts = "
                f"{contact_tensor[frame, 0, active_links]}"
            )

        # Real-time playback
        elapsed = time.perf_counter() - start_time
        remaining = frame_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    # SAVE CONTACT DATA
    output_dir = ep.episode_root / "processed"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    contact_file = output_dir / "contact_tensor.npy"

    mask_file = output_dir / "validity_mask.npy"

    np.save(
        contact_file,
        contact_tensor,
    )

    np.save(
        mask_file,
        validity_mask,
    )

    print("\n========== CONTACT RESULTS ==========")
    print("Contact tensor :", contact_tensor.shape)
    print("Validity mask  :", validity_mask.shape)
    print(
        "Frames with contact :",
        int(np.any(validity_mask[:, 0], axis=1).sum()),
        "/",
        timeline_len,
    )
    print(
        "Total active link contacts :",
        int(validity_mask.sum()),
    )
    print("Saved:", contact_file)
    print("Saved:", mask_file)
    print("=====================================\n")

    print("\nReplay finished.")


if __name__ == "__main__":
    main()
