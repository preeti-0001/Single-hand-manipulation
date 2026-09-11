from __future__ import annotations

import argparse
import time
from pathlib import Path

import genesis as gs
import numpy as np
import torch

from src.utils.common import (
    load_object_trajectory,
    load_robot_qpos_on_video_timeline,
    resolve_episode,
)
from src.utils.math_utils import matrix_to_wxyz

from .actor_critic import Actor
from .ppo import _apply_virtual_object_controller
from .rewards import RewardModule


DATASET_ROOT = Path("hrdexdb")
FPS = 30.0


def parse_args():
    p = argparse.ArgumentParser("Visualize a saved PPO + VOC checkpoint")
    p.add_argument("--hand", required=True)
    p.add_argument("--object_name", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--mode",
        choices=["actions", "policy"],
        default="actions",
        help="actions=replay saved best_actions; policy=run saved Actor.",
    )
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--action_scale", type=float, default=0.1)
    p.add_argument(
        "--use_voc",
        action="store_true",
        help="Apply approximate VOC during visualization.",
    )
    p.add_argument("--voc_strength", type=float, default=1.0)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="In policy mode, use Actor mean instead of sampling.",
    )
    return p.parse_args()


def load_data(args):
    ep = resolve_episode(
        dataset_root=DATASET_ROOT,
        hand=args.hand,
        object_name=args.object_name,
        scene=args.scene,
    )

    qpos, _, _, _, _ = load_robot_qpos_on_video_timeline(
        ep.episode_root,
        ep.hand,
    )

    object_poses = load_object_trajectory(
        DATASET_ROOT,
        ep.hand,
        ep.object_name,
        ep.scene,
    )

    timeline_len = min(len(qpos), len(object_poses))

    return ep, qpos, object_poses, timeline_len


def build_scene(ep, object_poses):
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

    T0 = object_poses[0]

    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=str(ep.object_mesh),
            pos=T0[:3, 3],
            quat=matrix_to_wxyz(T0[:3, :3]),
            scale=1.0,
            fixed=False,
        ),
        material=gs.materials.Rigid(friction=1.5),
    )

    scene.build(n_envs=1, env_spacing=(1.0, 1.0))

    return scene, robot, obj


def make_demo_tensors(qpos, object_poses, timeline_len, device, dtype):
    demo_robot_qpos = torch.as_tensor(
        qpos[:timeline_len], device=device, dtype=dtype
    )

    demo_object_positions = torch.as_tensor(
        np.asarray([T[:3, 3] for T in object_poses[:timeline_len]]),
        device=device,
        dtype=dtype,
    )

    demo_object_quaternions = torch.as_tensor(
        np.asarray(
            [matrix_to_wxyz(T[:3, :3]) for T in object_poses[:timeline_len]]
        ),
        device=device,
        dtype=dtype,
    )

    return (
        demo_robot_qpos,
        demo_object_positions,
        demo_object_quaternions,
    )


def robot_indices(robot):
    return [
        robot.get_joint(j.name).dofs_idx_local[0]
        for j in robot.joints
    ]


def reset_scene(
    robot,
    obj,
    dofs,
    demo_robot_qpos,
    demo_object_positions,
    demo_object_quaternions,
):
    robot.set_dofs_position(
        demo_robot_qpos[0].unsqueeze(0),
        dofs,
        zero_velocity=True,
    )
    obj.set_pos(
        demo_object_positions[0].unsqueeze(0),
        zero_velocity=True,
    )
    obj.set_quat(
        demo_object_quaternions[0].unsqueeze(0),
        zero_velocity=True,
    )


def apply_robot_action(robot, dofs, raw_action, action_scale):
    qpos = robot.get_qpos(qs_idx_local=dofs)

    qmin, qmax = robot.get_dofs_limit(dofs_idx_local=dofs)
    qmin = qmin.to(qpos.device, qpos.dtype)
    qmax = qmax.to(qpos.device, qpos.dtype)

    target = qpos + action_scale * raw_action
    target = torch.maximum(target, qmin)
    target = torch.minimum(target, qmax)

    robot.control_dofs_position(
        target,
        dofs_idx_local=dofs,
    )


def apply_voc(
    scene,
    obj,
    frame,
    demo_object_positions,
    demo_object_quaternions,
    strength,
):
    current_pos = obj.get_pos()
    current_quat = obj.get_quat()
    current_vel = obj.get_links_vel(
        links_idx_local=[0]
    )[:, 0, :]

    target_pos = demo_object_positions[frame + 1].unsqueeze(0)
    target_quat = demo_object_quaternions[frame + 1].unsqueeze(0)

    target_vel = (
        demo_object_positions[frame + 1]
        - demo_object_positions[frame]
    ) / (1.0 / FPS)
    target_vel = target_vel.unsqueeze(0)

    strength_tensor = torch.full(
        (current_pos.shape[0],),
        strength,
        device=current_pos.device,
        dtype=current_pos.dtype,
    )

    _apply_virtual_object_controller(
        obj=obj,
        scene=scene,
        current_pos=current_pos,
        current_quat=current_quat,
        current_lin_vel=current_vel,
        target_pos=target_pos,
        target_quat=target_quat,
        target_lin_vel=target_vel,
        voc_strength=strength_tensor,
    )


def replay_saved_actions(
    checkpoint,
    scene,
    robot,
    obj,
    dofs,
    demo_robot_qpos,
    demo_object_positions,
    demo_object_quaternions,
    action_scale,
    speed,
    use_voc,
    voc_strength,
):
    if "best_actions" not in checkpoint:
        raise KeyError(
            "Checkpoint has no 'best_actions'. "
            "Use a checkpoint produced by ppo(8).py."
        )

    actions = checkpoint["best_actions"]
    if not torch.is_tensor(actions):
        actions = torch.as_tensor(actions)

    actions = actions.to(
        device=robot.get_qpos().device,
        dtype=robot.get_qpos().dtype,
    )

    n = min(actions.shape[0], demo_robot_qpos.shape[0] - 1)

    print(f"Replaying {n} saved actions")
    print(f"Saved training reward: {checkpoint.get('reward', 'unknown')}")
    print(f"VOC in replay: {use_voc}")
    print(
        "Note: saved checkpoint does not contain per-frame VOC strength, "
        "so --use_voc is an approximate replay."
    )

    delay = 1.0 / FPS / max(speed, 1e-6)

    for frame in range(n):
        apply_robot_action(
            robot, dofs, actions[frame], action_scale
        )

        if use_voc:
            apply_voc(
                scene,
                obj,
                frame,
                demo_object_positions,
                demo_object_quaternions,
                voc_strength,
            )

        scene.step()

        print(
            f"Frame {frame + 1:04d}/{n} | "
            f"object={obj.get_pos()[0].detach().cpu().numpy()}"
        )

        time.sleep(delay)

    print("Replay finished. Close the Genesis viewer when finished.")


def policy_rollout(
    checkpoint,
    scene,
    robot,
    obj,
    dofs,
    timeline_len,
    demo_robot_qpos,
    demo_object_positions,
    demo_object_quaternions,
    action_scale,
    speed,
    use_voc,
    voc_strength,
    deterministic,
):
    if "actor_state_dict" not in checkpoint:
        raise KeyError("Checkpoint has no 'actor_state_dict'.")

    actor = Actor(robot_dof=len(dofs)).to(robot.get_qpos().device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    qmin, qmax = robot.get_dofs_limit(dofs_idx_local=dofs)
    qmin = qmin.to(robot.get_qpos().device, robot.get_qpos().dtype)
    qmax = qmax.to(robot.get_qpos().device, robot.get_qpos().dtype)

    delay = 1.0 / FPS / max(speed, 1e-6)

    print(f"Running learned policy for {timeline_len - 1} frames")
    print(f"Deterministic: {deterministic}")
    print(f"VOC in rollout: {use_voc}")

    with torch.no_grad():
        for frame in range(timeline_len - 1):
            robot_qpos = robot.get_qpos(qs_idx_local=dofs)
            object_pos = obj.get_pos()
            object_quat = obj.get_quat()

            target_robot_qpos = demo_robot_qpos[frame + 1].unsqueeze(0)
            target_object_pos = demo_object_positions[frame + 1].unsqueeze(0)
            target_object_quat = demo_object_quaternions[frame + 1].unsqueeze(0)

            dist = actor(
                robot_qpos,
                object_pos,
                object_quat,
                target_robot_qpos,
                target_object_pos,
                target_object_quat,
            )

            raw_action = (
                dist.mean if deterministic else dist.sample()
            )

            target_qpos = robot_qpos + action_scale * raw_action
            target_qpos = torch.maximum(target_qpos, qmin)
            target_qpos = torch.minimum(target_qpos, qmax)

            robot.control_dofs_position(
                target_qpos,
                dofs_idx_local=dofs,
            )

            if use_voc:
                apply_voc(
                    scene,
                    obj,
                    frame,
                    demo_object_positions,
                    demo_object_quaternions,
                    voc_strength,
                )

            scene.step()

            print(
                f"Frame {frame + 1:04d}/{timeline_len - 1} | "
                f"object={obj.get_pos()[0].detach().cpu().numpy()}"
            )

            time.sleep(delay)

    print("Policy rollout finished. Close the Genesis viewer when finished.")


def main():
    args = parse_args()

    if args.speed <= 0:
        raise ValueError("--speed must be > 0")

    ep, qpos, object_poses, timeline_len = load_data(args)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    print("\n========== CHECKPOINT ==========")
    print("Path   :", checkpoint_path)
    print("Episode:", checkpoint.get("episode", "unknown"))
    print("Env    :", checkpoint.get("best_env", "unknown"))
    print("Reward :", checkpoint.get("reward", "unknown"))
    print("VOC cfg:", checkpoint.get("voc_config", "not stored"))
    print("================================\n")

    scene, robot, obj = build_scene(ep, object_poses)

    dofs = robot_indices(robot)
    if len(dofs) != qpos.shape[1]:
        raise RuntimeError(
            f"DOF mismatch: Genesis={len(dofs)}, HRDexDB={qpos.shape[1]}"
        )

    device = robot.get_qpos().device
    dtype = robot.get_qpos().dtype

    (
        demo_robot_qpos,
        demo_object_positions,
        demo_object_quaternions,
    ) = make_demo_tensors(
        qpos,
        object_poses,
        timeline_len,
        device,
        dtype,
    )

    reset_scene(
        robot,
        obj,
        dofs,
        demo_robot_qpos,
        demo_object_positions,
        demo_object_quaternions,
    )

    if args.mode == "actions":
        replay_saved_actions(
            checkpoint,
            scene,
            robot,
            obj,
            dofs,
            demo_robot_qpos,
            demo_object_positions,
            demo_object_quaternions,
            args.action_scale,
            args.speed,
            args.use_voc,
            args.voc_strength,
        )
    else:
        policy_rollout(
            checkpoint,
            scene,
            robot,
            obj,
            dofs,
            timeline_len,
            demo_robot_qpos,
            demo_object_positions,
            demo_object_quaternions,
            args.action_scale,
            args.speed,
            args.use_voc,
            args.voc_strength,
            args.deterministic,
        )


if __name__ == "__main__":
    main()
