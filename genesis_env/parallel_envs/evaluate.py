from __future__ import annotations

import argparse
import time
from pathlib import Path

import genesis as gs
import torch

from src.utils.common import (
    load_robot_qpos_on_video_timeline,
    load_object_trajectory,
    resolve_episode,
)
from src.utils.math_utils import matrix_to_wxyz


DATASET_ROOT = Path("hrdexdb")
FPS = 30.0
ACTION_SCALE = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the best PPO trajectory saved in best_ppo.pt."
    )
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to the PPO checkpoint (.pt).")
    parser.add_argument("--hand", required=True, type=str,
                        help="Hand name, e.g. allegro_v5.")
    parser.add_argument("--object_name", required=True, type=str,
                        help="Object name, e.g. apple.")
    parser.add_argument("--scene", required=True, type=str,
                        help="HRDexDB scene ID, e.g. 4.")
    parser.add_argument("--env", type=int, default=None,
                        help="Environment index. Default: best_env in checkpoint.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed. 1.0 = real time.")
    parser.add_argument("--action_scale", type=float, default=ACTION_SCALE,
                        help="Action-to-joint delta scale used during training.")
    return parser.parse_args()


def load_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if "best_actions" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'best_actions'. "
            "This evaluator expects the checkpoint produced by the updated PPO code."
        )

    return checkpoint


def select_best_actions(best_actions, env_id=None):
    actions = torch.as_tensor(best_actions).detach().cpu().float()

    if actions.ndim == 2:
        return actions, 0 if env_id is None else env_id

    if actions.ndim != 3:
        raise ValueError(
            f"Unexpected best_actions shape: {tuple(actions.shape)}. "
            "Expected [T,A] or [T,N,A]."
        )

    if env_id is None:
        raise ValueError(
            "best_actions contains multiple environments. "
            "The checkpoint must contain best_env or you must pass --env."
        )

    if not 0 <= env_id < actions.shape[1]:
        raise IndexError(
            f"Environment {env_id} is outside [0, {actions.shape[1] - 1}]."
        )

    return actions[:, env_id, :], env_id


def get_robot_dof_indices(robot):
    return [
        robot.get_joint(joint.name).dofs_idx_local[0]
        for joint in robot.joints
    ]


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)

    stored_best_env = checkpoint.get("best_env", None)
    if stored_best_env is not None:
        stored_best_env = int(
            stored_best_env.item()
            if torch.is_tensor(stored_best_env)
            else stored_best_env
        )

    env_id = args.env if args.env is not None else stored_best_env

    print("\n========== CHECKPOINT ==========")
    print("Checkpoint :", checkpoint_path)
    print("Episode    :", checkpoint.get("episode", "N/A"))
    print("Reward     :", checkpoint.get("reward", "N/A"))
    print("Best env   :", stored_best_env)
    print("================================\n")

    ep = resolve_episode(
        dataset_root=DATASET_ROOT,
        hand=args.hand,
        object_name=args.object_name,
        scene=args.scene,
    )

    qpos, video_time, frame_ids, hand_dof, arm_dof = (
        load_robot_qpos_on_video_timeline(
            ep.episode_root,
            ep.hand,
        )
    )

    object_poses = load_object_trajectory(
        DATASET_ROOT,
        args.hand,
        args.object_name,
        args.scene,
    )

    timeline_len = min(len(qpos), len(object_poses))

    best_actions, selected_env = select_best_actions(
        checkpoint["best_actions"],
        env_id,
    )

    replay_len = min(timeline_len - 1, best_actions.shape[0])

    if replay_len <= 0:
        raise RuntimeError("No actions available for visualization.")

    best_actions = best_actions[:replay_len]

    print("========== TRAJECTORY ==========")
    print(
        "Stored best_actions :",
        tuple(torch.as_tensor(checkpoint["best_actions"]).shape),
    )
    print("Selected env        :", selected_env)
    print("Actions replayed    :", tuple(best_actions.shape))
    print("Demo frames         :", timeline_len)
    print("Playback frames     :", replay_len)
    print("Action scale        :", args.action_scale)
    print("================================\n")

    gs.init(
        backend=gs.gpu,
        logging_level="warning",
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

    scene.add_entity(gs.morphs.Plane())

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(ep.robot_urdf),
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            recompute_inertia=True,
        )
    )

    first_pose = object_poses[0]
    first_position = first_pose[:3, 3]
    first_quat = matrix_to_wxyz(first_pose[:3, :3])

    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=str(ep.object_mesh),
            pos=first_position,
            quat=first_quat,
            scale=1.0,
            fixed=False,
        ),
        material=gs.materials.Rigid(friction=1.5),
    )

    scene.build()

    motors_dof_idx = get_robot_dof_indices(robot)

    print("\n========== ROBOT ==========")
    print("Genesis qpos size :", robot.n_qs)
    print("Genesis DOFs      :", robot.n_dofs)
    print("HRDexDB qpos size :", qpos.shape[1])
    print("Action dimension   :", len(motors_dof_idx))
    print("============================\n")

    if robot.n_qs != qpos.shape[1]:
        raise RuntimeError(
            "DOF mismatch!\n"
            f"Genesis qpos : {robot.n_qs}\n"
            f"HRDexDB qpos : {qpos.shape[1]}"
        )

    if best_actions.shape[-1] != len(motors_dof_idx):
        raise RuntimeError(
            "Action dimension mismatch!\n"
            f"Checkpoint : {best_actions.shape[-1]}\n"
            f"Robot      : {len(motors_dof_idx)}"
        )

    initial_qpos = torch.as_tensor(
        qpos[0], dtype=torch.float32, device=gs.device
    )
    initial_position = torch.as_tensor(
        first_position, dtype=torch.float32, device=gs.device
    )
    initial_quat = torch.as_tensor(
        first_quat, dtype=torch.float32, device=gs.device
    )

    robot.set_dofs_position(
        initial_qpos,
        motors_dof_idx,
        zero_velocity=True,
    )
    obj.set_pos(initial_position, zero_velocity=True)
    obj.set_quat(initial_quat, zero_velocity=True)

    scene.step()

    print("==============================================")
    print("Starting PPO trajectory visualization")
    print("Close the Genesis viewer to finish.")
    print("==============================================\n")

    action_tensor = best_actions.to(gs.device)
    frame_dt = 1.0 / FPS

    for frame in range(replay_len):
        start = time.perf_counter()

        current_qpos = robot.get_qpos(
            qs_idx_local=motors_dof_idx
        )

        raw_action = action_tensor[frame]
        delta_q = raw_action * args.action_scale
        target_qpos = current_qpos + delta_q

        lower, upper = robot.get_dofs_limit(
            dofs_idx_local=motors_dof_idx
        )
        target_qpos = torch.clamp(
            target_qpos,
            lower,
            upper,
        )

        robot.control_dofs_position(
            target_qpos,
            dofs_idx_local=motors_dof_idx,
        )

        scene.step()

        if frame % 30 == 0:
            object_position = obj.get_pos()
            print(
                f"Frame {frame:4d}/{replay_len} | "
                f"object = "
                f"{object_position[0].detach().cpu().numpy()}"
            )

        elapsed = time.perf_counter() - start
        sleep_time = frame_dt / max(args.speed, 1e-6) - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

    print("\nVisualization finished.")


if __name__ == "__main__":
    main()
