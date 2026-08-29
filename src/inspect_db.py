import numpy as np
from pathlib import Path

OBJECT_NAME = "apple"  # Replace with the actual object name
HAND="allegro_v5"
SCENE="4"
episode_path = Path(f"hrdexdb/{HAND}/{OBJECT_NAME}/{SCENE}")

for object_trajectory in episode_path.rglob("*.npy"):
    print("\nFILE:", object_trajectory)

    try:
        data = np.load(object_trajectory, allow_pickle=True)

        print("type :", type(data))
        print("shape:", getattr(data, "shape", None))
        print("dtype:", getattr(data, "dtype", None))

        if hasattr(data, "files"):
            print("keys :", data.files)

    except Exception as e:
        print("ERROR:", e)


object_trajectory = f"hrdexdb/object_6d_pose_v2/{HAND}/{OBJECT_NAME}_{SCENE}.npz"

object_trajectory_data = np.load(object_trajectory)

print("NPZ FILE:", object_trajectory)
print("Keys:", object_trajectory_data.files)

for key in object_trajectory_data.files:
    print(
        key,
        "shape =", object_trajectory_data[key].shape,
        "dtype =", object_trajectory_data[key].dtype
    )
    break


hand_base = f"hrdexdb/{HAND}/{OBJECT_NAME}/{SCENE}/raw/hand"

position = np.load(f"{hand_base}/position.npy")
action = np.load(f"{hand_base}/action.npy")
time = np.load(f"{hand_base}/time.npy")
tactile = np.load(f"{hand_base}/tactile.npy")

print("position:", position.shape)
print("action:", action.shape)
print("time:", time.shape)
print("tactile:", tactile.shape)

print("\nFirst hand position:")
print(position[0])

print("\nLast hand position:")
print(position[-1])

print("\nFirst time:", time[0])
print("Last time:", time[-1])

print("\nTime differences:")
print(np.min(np.diff(time)))
print(np.max(np.diff(time)))
print(np.mean(np.diff(time)))