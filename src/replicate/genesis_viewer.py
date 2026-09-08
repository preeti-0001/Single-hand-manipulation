from __future__ import annotations

import time
from pathlib import Path

import genesis as gs
import numpy as np

from src.utils.common import (
    resolve_episode,
    load_robot_qpos_on_video_timeline,
    load_human_mano_sequence,
    load_object_trajectory
)

from src.utils.math_utils import matrix_to_wxyz
import argparse


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("hrdexdb")

FPS = 30.0


# ============================================================
# LOAD OBJECT NPZ
# ============================================================

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

# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # Resolve HRDexDB episode
    # ========================================================
    
    args = parse_args()

    HAND = args.hand
    OBJECT_NAME = args.object_name
    SCENE = args.scene

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


    # ========================================================
    # Robot trajectory
    #
    # EXACTLY from common.py
    # ========================================================
    
    if ep.hand == "human":
        mano_vertices, mano_faces, frame_ids, video_time = load_human_mano_sequence(ep.episode_root)
        timeline_len = len(mano_vertices)
        qpos = None
    else:
        qpos, video_time, frame_ids, hand_dof, arm_dof = load_robot_qpos_on_video_timeline(ep.episode_root, ep.hand)
        timeline_len = len(qpos)
    
    if timeline_len <= 0:
        raise ValueError(f"Empty trajectory: {ep.episode_root}")

    print("Robot qpos shape :", qpos.shape)
    print("Video time shape :", video_time.shape)
    print("Frame IDs shape  :", frame_ids.shape)


    # ========================================================
    # Object trajectory
    #
    # Your working NPZ loader
    # ========================================================

    object_poses = load_object_trajectory(
        DATASET_ROOT,
        HAND,
        OBJECT_NAME,
        SCENE,
    )


    # ========================================================
    # Determine replay length
    # ========================================================

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


    # ========================================================
    # Genesis
    # ========================================================

    gs.init(
        backend=gs.gpu
    )

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


    # ========================================================
    # Ground
    # ========================================================

    plane = scene.add_entity(
        gs.morphs.Plane()
    )


    # ========================================================
    # Robot
    # ========================================================

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(ep.robot_urdf),

            pos=(0.0, 0.0, 0.0),

            fixed=True,

            # Required for your current URDF.
            recompute_inertia=True,
        )
    )


    # ========================================================
    # Object
    # ========================================================

    first_pose = object_poses[0]

    first_position = first_pose[:3, 3]

    first_quat = matrix_to_wxyz(
        first_pose[:3, :3]
    )

    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=str(ep.object_mesh),

            pos=first_position,

            quat=first_quat,

            scale=1.0,

            # Kinematic replay.
            fixed=True,
        )
    )


    # ========================================================
    # Build
    # ========================================================

    scene.build()


    # ========================================================
    # Robot DOF check
    # ========================================================

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


    # ========================================================
    # Print joints
    # ========================================================

    print("Genesis joints:")

    for i, joint in enumerate(robot.joints):

        print(
            f"{i:02d} | {joint.name}"
        )


    # ========================================================
    # Replay
    # ========================================================

    print(
        f"\nStarting replay "
        f"({timeline_len} frames @ {FPS} FPS)\n"
    )


    frame_dt = 1.0 / FPS


    for frame in range(timeline_len):

        start_time = time.perf_counter()


        # ====================================================
        # ROBOT
        # ====================================================

        robot.set_qpos(
            qpos[frame],
            zero_velocity=True,
        )


        # ====================================================
        # OBJECT
        #
        # This is directly based on your working
        # object-only Genesis code.
        # ====================================================

        T = object_poses[frame]


        # ----------------------------------------------------
        # Position
        # ----------------------------------------------------

        position = T[:3, 3]


        # ----------------------------------------------------
        # Rotation
        # ----------------------------------------------------

        rotation_matrix = T[:3, :3]

        quat_wxyz = matrix_to_wxyz(
            rotation_matrix
        )


        # ----------------------------------------------------
        # Move object
        # ----------------------------------------------------

        obj.set_pos(
            position,
            zero_velocity=True,
        )

        obj.set_quat(
            quat_wxyz,
            zero_velocity=True,
        )


        # ====================================================
        # Advance Genesis
        # ====================================================

        scene.step()


        # ====================================================
        # Real-time playback
        # ====================================================

        elapsed = (
            time.perf_counter()
            - start_time
        )

        remaining = frame_dt - elapsed

        if remaining > 0:
            time.sleep(remaining)


        # ====================================================
        # Debug
        # ====================================================

        print(
            f"Frame {frame:04d} | "
            f"Object pos = {position} | "
            f"Robot qpos = {qpos[frame]}"
        )


    print("\nReplay finished.")


if __name__ == "__main__":
    main()