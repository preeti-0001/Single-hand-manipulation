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
# LOAD OBJECT NPZ
# ============================================================

def load_object_trajectory(
    dataset_root: Path,
    hand: str,
    object_name: str,
    scene: str,
):
    episode_root = (
        dataset_root
        / hand
        / object_name
        / scene
    )

    pose_file = episode_root / "object_6d_pose.npz"

    if not pose_file.exists():
        raise FileNotFoundError(
            f"Object pose file not found:\n{pose_file}"
        )

    data = np.load(pose_file)

    print("\n========== OBJECT DATA ==========")
    print("Pose file:", pose_file)
    print("Keys:", data.files)

    frame_keys = sorted(
        data.files,
        key=lambda x: int(x.split("_")[1]),
    )

    poses_world = np.asarray(
        [data[key] for key in frame_keys],
        dtype=float,
    )

    print("Raw poses:", poses_world.shape)

    # ========================================================
    # IMPORTANT:
    # Convert HRDexDB world frame → robot frame
    #
    # This is the SAME transformation used by common.py
    # ========================================================

    c2r = load_c2r(episode_root)

    robot_from_world = np.linalg.inv(c2r)

    poses_robot = np.einsum(
        "ij,tjk->tik",
        robot_from_world,
        poses_world,
    )

    print("C2R:")
    print(c2r)

    print("\nRobot-from-world:")
    print(robot_from_world)

    print("\nFirst raw pose:")
    print(poses_world[0])

    print("\nFirst robot-frame pose:")
    print(poses_robot[0])

    print("\nLast robot-frame pose:")
    print(poses_robot[-1])

    print("=================================\n")

    return poses_robot
# ============================================================
# QUATERNION
# ============================================================

def matrix_to_wxyz(rotation_matrix):

    quat_xyzw = Rotation.from_matrix(
        rotation_matrix
    ).as_quat()

    return np.array(
        [
            quat_xyzw[3],
            quat_xyzw[0],
            quat_xyzw[1],
            quat_xyzw[2],
        ],
        dtype=float,
    )


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

    qpos, video_time, frame_ids = (
        load_robot_qpos_on_video_timeline(
            ep.episode_root,
            ep.hand,
        )
    )

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

            # Kinematic replay false.
            fixed=False,

        ),
        material=gs.materials.Rigid(
            friction=1.5,
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

        # check contacts (grasping)
        contacts = robot.get_contacts(with_entity=obj)

        print(contacts)


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

        # print(
        #     f"Frame {frame:04d} | "
        #     f"Object pos = {position} | "
        #     f"Robot qpos = {qpos[frame]}"
        # )


    print("\nReplay finished.")


if __name__ == "__main__":
    main()