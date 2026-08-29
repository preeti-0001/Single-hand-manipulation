import numpy as np

position = np.load(
    "hrdexdb/allegro_v5/apple/4/raw/hand/position.npy"
)

action = np.load(
    "hrdexdb/allegro_v5/apple/4/raw/hand/action.npy"
)

delta = action - position

for j in range(16):
    print(
        j,
        "mean:", delta[:, j].mean(),
        "std :", delta[:, j].std(),
        "min :", delta[:, j].min(),
        "max :", delta[:, j].max(),
    )

base = "hrdexdb/allegro_v5/apple/4/raw/hand/"
base = ""

position = np.load(f"{base}right_commands_time.npy")
action = np.load(f"{base}right_commands.npy")

print("Temporal correlation")
print(action.shape)
print(action[0])
