import argparse
import os
import numpy as np
import trimesh


def inspect(name, path, allow_pickle=False):
    """Load and print basic information about an .npy/.npz file."""
    if not os.path.exists(path):
        print(f"\n[NOT FOUND] {path}")
        return

    try:
        x = np.load(path, allow_pickle=allow_pickle)

        print(f"\n{'=' * 70}")
        print(name)
        print(f"{'=' * 70}")
        print("path :", path)

        if isinstance(x, np.lib.npyio.NpzFile):
            print("type :", "NPZ")
            print("keys :", x.files)

            for key in x.files:
                arr = x[key]
                print(f"\nKEY: {key}")
                print("  shape :", arr.shape)
                print("  dtype :", arr.dtype)

                print("  min   :", arr.min())
                print("  max   :", arr.max())

                if arr.ndim > 0:
                    print("  first :", arr[0])
                    print("  last  :", arr[-1])
                else:
                    print("  value :", arr)

            x.close()

        else:
            print("type  :", "NPY")
            print("shape :", x.shape)
            print("dtype :", x.dtype)
            print("min   :", x.min())
            print("max   :", x.max())

            if x.ndim > 0:
                print("first :", x[0])
                print("last  :", x[-1])
            else:
                print("value :", x)

    except Exception as e:
        print(f"[ERROR] Could not inspect {path}")
        print("       ", e)


def inspect_hand(base):
    """Inspect raw hand data."""
    print("\n" + "#" * 80)
    print("# HAND")
    print("#" * 80)

    hand_base = os.path.join(base, "raw", "hand")

    for name in ["position", "action", "tactile", "time"]:
        inspect(
            f"HAND {name.upper()}",
            os.path.join(hand_base, f"{name}.npy"),
            allow_pickle=True,
        )

def inspect_arm(base):
    """Inspect raw arm data."""
    print("\n" + "#" * 80)
    print("# ARM")
    print("#" * 80)

    arm_base = os.path.join(base, "raw", "arm")

    for name in ["position", "action", "torque", "time", "velocity"]:
        inspect(
            f"ARM {name.upper()}",
            os.path.join(arm_base, f"{name}.npy"),
            allow_pickle=True,
        )

def inspect_processed(base):
    """Inspect processed DexMachina-related data."""
    print("\n" + "#" * 80)
    print("# PROCESSED")
    print("#" * 80)

    processed_base = os.path.join(base, "processed")

    for name in ["contact_tensor", "validity_mask", "optimized_contact_tensor", "optimized_validity_mask"]:
        inspect(
            f"PROCESSED {name.upper()}",
            os.path.join(processed_base, f"{name}.npy"),
        )
    for name in ["grasp_candidates", "ideal_grasps"]:
        inspect(
            f"PROCESSED {name.upper()}",
            os.path.join(processed_base, f"{name}.npy"),
            allow_pickle=True,
        )


def inspect_object(base, hand, object_name, scene):
    """Inspect object pose data and object mesh."""
    print("\n" + "#" * 80)
    print("# OBJECT")
    print("#" * 80)

    # ------------------------------------------------------------------
    # Object 6D pose inside episode
    # ------------------------------------------------------------------

    pose_path = os.path.join(base, "object_6d_pose.npz")

    inspect(
        "OBJECT 6D POSE",
        pose_path,
    )

    # ------------------------------------------------------------------
    # Object 6D pose V2
    # ------------------------------------------------------------------

    pose_v2_path = os.path.join(
        "hrdexdb",
        "object_6d_pose_v2",
        hand,
        f"{object_name}_{scene}.npz",
    )

    inspect(
        "OBJECT 6D POSE V2",
        pose_v2_path,
    )

    # ------------------------------------------------------------------
    # Object mesh
    # ------------------------------------------------------------------

    mesh_path = os.path.join(
        "hrdexdb",
        "assets",
        "mesh",
        object_name,
        f"{object_name}.obj",
    )

    print(f"\n{'=' * 70}")
    print("OBJECT MESH")
    print(f"{'=' * 70}")
    print("path :", mesh_path)

    if not os.path.exists(mesh_path):
        print("[NOT FOUND]")
        return

    try:
        mesh = trimesh.load(mesh_path, force="mesh")

        print("vertices :", len(mesh.vertices))
        print("faces    :", len(mesh.faces))
        print("bounds   :")
        print(mesh.bounds)
        print("size     :", mesh.extents)

    except Exception as e:
        print("[ERROR] Could not load mesh:")
        print("       ", e)


def inspect_camera(base):
    """Inspect camera calibration / C2R information."""
    print("\n" + "#" * 80)
    print("# CAMERA")
    print("#" * 80)

    c2r_path = os.path.join(base, "C2R.npy")

    inspect(
        "CAMERA C2R",
        c2r_path,
    )
    print("\nCAMERA C2R matrix:")
    c2r = np.load(c2r_path)
    print(c2r)


def inspect_timestamps(base):
    """Inspect arm/hand timestamps and frame timestamps."""
    print("\n" + "#" * 80)
    print("# TIMESTAMPS")
    print("#" * 80)

    # ------------------------------------------------------------------
    # ARM TIME
    # ------------------------------------------------------------------

    arm_time_path = os.path.join(
        base,
        "raw",
        "arm",
        "time.npy",
    )

    if os.path.exists(arm_time_path):
        arm_time = np.load(
            arm_time_path,
            allow_pickle=True,
        ).astype(np.float64)

        print("\nARM TIME")
        print("length   :", len(arm_time))
        print("first    :", arm_time[0])
        print("last     :", arm_time[-1])
        print("duration :", arm_time[-1] - arm_time[0])

    # ------------------------------------------------------------------
    # HAND TIME
    # ------------------------------------------------------------------

    hand_time_path = os.path.join(
        base,
        "raw",
        "hand",
        "time.npy",
    )

    if os.path.exists(hand_time_path):
        hand_time = np.load(
            hand_time_path,
            allow_pickle=True,
        ).astype(np.float64)

        print("\nHAND TIME")
        print("length   :", len(hand_time))
        print("first    :", hand_time[0])
        print("last     :", hand_time[-1])
        print("duration :", hand_time[-1] - hand_time[0])

    # ------------------------------------------------------------------
    # FRAME ID
    # ------------------------------------------------------------------

    frame_id_path = os.path.join(
        base,
        "raw",
        "timestamps",
        "frame_id.npy",
    )

    if os.path.exists(frame_id_path):
        frame_id = np.load(
            frame_id_path,
            allow_pickle=True,
        )

        print("\nFRAME ID")
        print("shape :", frame_id.shape)
        print("first 20 :", frame_id[:20])
        print("last 20  :", frame_id[-20:])

    # ------------------------------------------------------------------
    # TIMESTAMP
    # ------------------------------------------------------------------

    timestamp_path = os.path.join(
        base,
        "raw",
        "timestamps",
        "timestamp.npy",
    )

    if os.path.exists(timestamp_path):
        timestamp = np.load(
            timestamp_path,
            allow_pickle=True,
        ).astype(np.float64)

        print("\nTIMESTAMP")
        print("shape    :", timestamp.shape)
        print("first 20 :", timestamp[:20])
        print("last 20  :", timestamp[-20:])
        print("duration :", timestamp[-1] - timestamp[0])


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inspect HRDexDB episode data."
    )

    # ------------------------------------------------------------------
    # Episode identification
    # ------------------------------------------------------------------

    parser.add_argument(
        "--hand",
        required=True,
        help="Hand name, e.g. allegro_v5",
    )

    parser.add_argument(
        "--object_name",
        required=True,
        help="Object name, e.g. apple",
    )

    parser.add_argument(
        "--scene",
        required=True,
        help="Scene ID, e.g. 4",
    )

    # ------------------------------------------------------------------
    # Component filters
    # ------------------------------------------------------------------

    parser.add_argument(
        "--only_hand",
        action="store_true",
        help="Inspect only raw hand data.",
    )

    parser.add_argument(
        "--only_processed",
        action="store_true",
        help="Inspect only processed data.",
    )

    parser.add_argument(
        "--only_object",
        action="store_true",
        help="Inspect only object pose and mesh data.",
    )

    parser.add_argument(
        "--only_camera",
        action="store_true",
        help="Inspect only camera C2R data.",
    )

    parser.add_argument(
        "--only_timestamps",
        action="store_true",
        help="Inspect only timestamp/frame data.",
    )

    parser.add_argument(
        "--only_arm",
        action="store_true",
        help="Inspect only arm data.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    base = os.path.join(
        "hrdexdb",
        args.hand,
        args.object_name,
        args.scene,
    )

    print("\n" + "=" * 80)
    print("HRDexDB INSPECTOR")
    print("=" * 80)

    print("Hand   :", args.hand)
    print("Object :", args.object_name)
    print("Scene  :", args.scene)
    print("Base   :", base)

    if not os.path.exists(base):
        print(f"\n[ERROR] Episode directory does not exist:")
        print(f"       {base}")
        return

    # --------------------------------------------------------------
    # Determine whether any --only_* flag was supplied
    # --------------------------------------------------------------

    only_flags = [
        args.only_hand,
        args.only_processed,
        args.only_object,
        args.only_camera,
        args.only_timestamps,
        args.only_arm,
    ]

    inspect_everything = not any(only_flags)

    # --------------------------------------------------------------
    # Select components
    # --------------------------------------------------------------

    if inspect_everything or args.only_hand:
        inspect_hand(base)

    if inspect_everything or args.only_arm:
        inspect_arm(base)

    if inspect_everything or args.only_processed:
        inspect_processed(base)

    if inspect_everything or args.only_object:
        inspect_object(
            base,
            args.hand,
            args.object_name,
            args.scene,
        )

    if inspect_everything or args.only_camera:
        inspect_camera(base)

    if inspect_everything or args.only_timestamps:
        inspect_timestamps(base)


if __name__ == "__main__":
    main()