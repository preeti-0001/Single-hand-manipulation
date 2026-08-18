
```
conda create -n genesis python=3.12 -y
conda activate genesis
pip install numpy scipy matplotlib trimesh
pip install genesis-world
pip install torch
export TORCH_COMPILE_DISABLE=1
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

```



```
HRDexDB recording #4
│
├── Allegro hand
│     ├── position.npy   → (1357, 16)
│     ├── action.npy     → (1357, 16)
│     └── time.npy       → (1357,)
│
├── Arm
│     ├── position.npy   → (1798, 6)
│     ├── action.npy     → (1798, 4, 4)
│     ├── velocity.npy   → (1798, 6)
│     ├── torque.npy     → (1798, 6)
│     └── time.npy       → (1798,)
│
├── Object
│     └── object_6d_pose.npz
│           └── frame_0 ... frame_406
│
├── C2R.npy
│     └── 4×4 coordinate transform
│
└── Camera timestamps
      ├── frame_id → 1 ... 407
      └── timestamp → 407 samples

```