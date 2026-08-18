from __future__ import annotations

import time
from pathlib import Path

import genesis as gs
import numpy as np
from scipy.spatial.transform import Rotation as R

from common import (
    resolve_episode,
    load_robot_qpos_on_video_timeline,
    load_object_poses_robot,
)


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("hrdexdb")

HAND = "allegro_v5"
OBJECT_NAME = "apple"
SCENE = "4"

FPS = 30.0


# ============================================================
# RESOLVE HRDexDB EPISODE
# ============================================================

ep = resolve_episode(
    dataset_root=DATASET_ROOT,
    hand=HAND,
    object_name=OBJECT_NAME,
    scene=SCENE,
)

print("\n========== HRDexDB EPISODE ==========")
print("Dataset root :", ep.dataset_root)
print("Hand         :", ep.hand)
print("Hand dir     :", ep.hand_dir)
print("Object       :", ep.object_name)
print("Scene        :", ep.scene)
print("Episode root :", ep.episode_root)
print("Object mesh  :", ep.object_mesh)
print("Robot URDF   :", ep.robot_urdf)
print("=====================================\n")


# ============================================================
# LOAD ROBOT TRAJECTORY
# ============================================================

qpos, video_time, frame_ids = load_robot_qpos_on_video_timeline(
    ep.episode_root,
    ep.hand,
)

print("Robot qpos shape :", qpos.shape)
print("Video time shape :", video_time.shape)
print("Frame IDs shape  :", frame_ids.shape)


# ============================================================
# LOAD OBJECT TRAJECTORY
#
# IMPORTANT:
# This uses your exact common.py implementation.
#
# Therefore:
#
# object_6d/pose_*.txt
#        ↓
# resample_poses()
#        ↓
# C2R transformation
#        ↓
# robot-frame object poses
# ============================================================

object_poses = load_object_poses_robot(
    ep.episode_root,
    target_len=len(qpos),
)

if object_poses is not None:
    print("Object pose shape :", object_poses.shape)
else:
    print("No object trajectory found.")


# ============================================================
# GENESIS INITIALIZATION
# ============================================================

gs.init()

scene = gs.Scene(
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

    show_viewer=True,
)


# ============================================================
# FLOOR
# ============================================================

scene.add_entity(
    gs.morphs.Plane()
)


# ============================================================
# ROBOT
# ============================================================

robot = scene.add_entity(
    gs.morphs.URDF(
        file=str(ep.robot_urdf),

        # HRDexDB robot coordinate frame starts here.
        pos=(0.0, 0.0, 0.0),

        # IMPORTANT:
        # We are replaying a recorded robot trajectory.
        fixed=True,
        recompute_inertia=True
    )
)


# ============================================================
# OBJECT
# ============================================================

obj = None

if object_poses is not None:

    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=str(ep.object_mesh),

            # Initial pose.
            pos=object_poses[0, :3, 3],

            # Initial orientation.
            quat=R.from_matrix(
                object_poses[0, :3, :3]
            ).as_quat()[[3, 0, 1, 2]],

            # Do NOT let the object fall during replay.
            fixed=True,
        )
    )


# ============================================================
# BUILD SCENE
# ============================================================

scene.build()


# ============================================================
# CHECK ROBOT DOFs
# ============================================================

print("\n========== GENESIS ROBOT ==========")
print("Genesis qpos size :", robot.n_qs)
print("Genesis DOFs      :", robot.n_dofs)
print("HRDexDB qpos size :", qpos.shape[1])
print("===================================\n")


if robot.n_qs != qpos.shape[1]:
    raise RuntimeError(
        "\nDOF mismatch!\n"
        f"Genesis robot qpos : {robot.n_qs}\n"
        f"HRDexDB trajectory : {qpos.shape[1]}\n\n"
        "This means the joint ordering/robot model does not "
        "match the HRDexDB trajectory."
    )


# ============================================================
# PRINT JOINT INFORMATION
# ============================================================

print("Genesis joints:")

for i, joint in enumerate(robot.joints):
    print(
        f"{i:02d} | "
        f"{joint.name}"
    )

print()


# ============================================================
# REPLAY FUNCTION
# ============================================================

def update_frame(frame: int):

    frame = int(
        np.clip(
            frame,
            0,
            len(qpos) - 1,
        )
    )

    # --------------------------------------------------------
    # ROBOT
    # --------------------------------------------------------

    robot.set_qpos(
        qpos[frame],
        zero_velocity=True,
    )


    # --------------------------------------------------------
    # OBJECT
    # --------------------------------------------------------

    if obj is not None:

        pose = object_poses[frame]

        position = pose[:3, 3]

        rotation_matrix = pose[:3, :3]

        quaternion_xyzw = R.from_matrix(
            rotation_matrix
        ).as_quat()

        # SciPy:
        #
        # x y z w
        #
        # Genesis:
        #
        # w x y z

        quaternion_wxyz = quaternion_xyzw[
            [3, 0, 1, 2]
        ]

        obj.set_pos(
            position,
            zero_velocity=True,
        )

        obj.set_quat(
            quaternion_wxyz,
            zero_velocity=True,
        )


# ============================================================
# INITIAL STATE
# ============================================================

update_frame(0)


# ============================================================
# REPLAY
# ============================================================

print(
    f"Starting replay: "
    f"{len(qpos)} frames @ {FPS} FPS"
)

print("Close the Genesis viewer to stop.\n")


frame_dt = 1.0 / FPS

for frame in range(len(qpos)):

    start_time = time.perf_counter()

    # Update recorded state.
    update_frame(frame)

    # Advance Genesis.
    scene.step()

    # Keep playback approximately real-time.
    elapsed = time.perf_counter() - start_time

    remaining = frame_dt - elapsed

    if remaining > 0:
        time.sleep(remaining)


print("Replay finished.")