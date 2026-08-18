import numpy as np
import os

base = "hrdexdb/allegro_v5/apple/4"


def inspect(name, path):
    x = np.load(path)
    print(f"\n{name}")
    print("shape:", x.shape)
    print("dtype:", x.dtype)
    print("first:", x[0] if x.ndim > 0 else x)
    print("last :", x[-1] if x.ndim > 0 else x)


inspect("HAND POSITION", f"{base}/raw/hand/position.npy")

inspect("HAND ACTION", f"{base}/raw/hand/action.npy")

inspect("HAND TIME", f"{base}/raw/hand/time.npy")

print("\nOBJECT 6D POSE")

pose = np.load(f"{base}/object_6d_pose.npz")

print("Object 6D pose keys:", pose.files)

for key in pose.files:
    x = pose[key]
    print(f"EXAMPLE")
    print(f"{key}: shape={x.shape}, dtype={x.dtype}")
    print("first:", x[0])
    break


print("\nC2R")

C2R = np.load(f"{base}/C2R.npy")

print("shape:", C2R.shape)
print(C2R)

print("\nARM")
abc=True

for name in ["position", "action", "velocity", "torque", "time"]:
    path = f"{base}/raw/arm/{name}.npy"
    print(path)
    x = np.load(path, allow_pickle=True)
    if abc:
        print(type(x))
        abc=False
    print(f"{name}: shape={x.shape}, dtype={x.dtype}, " f"first={x[0]}")

arm_time = np.load(
    f"{base}/raw/arm/time.npy",
    allow_pickle=True
).astype(np.float64)

hand_time = np.load(
    f"{base}/raw/hand/time.npy",
    allow_pickle=True
).astype(np.float64
)

frame_id = np.load(
    f"{base}/raw/timestamps/frame_id.npy",
    allow_pickle=True
)

timestamp = np.load(
    f"{base}/raw/timestamps/timestamp.npy",
    allow_pickle=True
).astype(np.float64)

print("ARM")
print(len(arm_time), arm_time[0], arm_time[-1])
print("duration:", arm_time[-1] - arm_time[0])

print("\nHAND")
print(len(hand_time), hand_time[0], hand_time[-1])
print("duration:", hand_time[-1] - hand_time[0])

print("\nFRAME ID")
print(frame_id.shape)
print(frame_id[:20])
print(frame_id[-20:])

print("\nTIMESTAMP")
print(timestamp.shape)
print(timestamp[:20])
print(timestamp[-20:])
print("duration:", timestamp[-1] - timestamp[0])


path = "hrdexdb/object_6d_pose_v2/allegro_v5/apple_4.npz"

data = np.load(path)

print("keys Object 6D pose V2:", data.files)

for key in data.files:

    x = data[key]
    print(f"EXAMPLE")
    print("\nKEY:", key)
    print("shape:", x.shape)
    print("dtype:", x.dtype)

    if x.ndim > 0:
        print("first:", x[0])
        print("last :", x[-1])
    break

import trimesh

mesh = trimesh.load(
    "hrdexdb/assets/mesh/apple/apple.obj",
    force="mesh"
)

print("vertices:", len(mesh.vertices))
print("faces:", len(mesh.faces))
print("bounds:")
print(mesh.bounds)

print("size:", mesh.extents)
