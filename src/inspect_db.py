import numpy as np
from pathlib import Path

object_name = "apple"  # Replace with the actual object name
hand_name="allegro_v5"
episode="4"
episode_path = Path(f"hrdexdb/{hand_name}/{object_name}/{episode}")

for path in episode_path.rglob("*.npy"):
    print("\nFILE:", path)

    try:
        data = np.load(path, allow_pickle=True)

        print("type :", type(data))
        print("shape:", getattr(data, "shape", None))
        print("dtype:", getattr(data, "dtype", None))

        if hasattr(data, "files"):
            print("keys :", data.files)

    except Exception as e:
        print("ERROR:", e)


path = f"hrdexdb/object_6d_pose_v2/{hand_name}/{object_name}_{episode}.npz"

data = np.load(path)

print("NPZ FILE:", path)
print("Keys:", data.files)

for key in data.files:
    print(
        key,
        "shape =", data[key].shape,
        "dtype =", data[key].dtype
    )
    break


base = f"hrdexdb/{hand_name}/{object_name}/{episode}/raw/hand"

position = np.load(f"{base}/position.npy")
action = np.load(f"{base}/action.npy")
time = np.load(f"{base}/time.npy")
tactile = np.load(f"{base}/tactile.npy")

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