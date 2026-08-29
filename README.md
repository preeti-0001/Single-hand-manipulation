
```
conda create -n genesis python=3.12 -y
conda activate genesis
pip install -r requirements.txt
```

export TORCH_COMPILE_DISABLE=1
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

```



```
hrdexdb\apple\raw
│
├── allegro_v5
│     ├── apple
│     │     ├──4 
├── hand
├── hand
├── hand
├── hand
├── hand
├── hand
│     ├── position.npy   → (1357, 16)
│     ├── action.npy     → (1357, 16)
│     └── time.npy       → (1357,)
│
├── arm
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
└── timestamps
      ├── frame_id → 1 ... 407
      └── timestamp → 407 samples

```

1. Replay the action in genesis
```
 python -m src.replicate.genesis_viewer  
```
2. Make contact tensor
```
python -m src.grasping.make_contact_tensor --hand allegro_v5 --object_name apple --scene 4 
```
3. optimize contact tensors
```
python -m src.grasping.contact_optimization --hand allegro_v5 --object_name apple --scene 4 
```
4. 
```
python -m src.grasping.grasping_without_modification
```
5. 
```
python -m src.inspect.dataset --hand HAND --object_name OBJECT_NAME --scene SCENE [--only_hand] [--only_processed] [--only_object] [--only_camera] [--only_timestamps]
example: 
python -m src.inspect.dataset --hand allegro_v5 --object_name apple --scene 4 --only_processed
```
6.
python -m train 
