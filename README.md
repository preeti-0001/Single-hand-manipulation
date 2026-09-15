```
conda create -n genesis python=3.12 -y
conda activate genesis
pip install -r requirements.txt
```

```
conda activate genesis
export TORCH_COMPILE_DISABLE=1
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

```

```

hrdexdb\apple\raw
│
├── allegro_v5
│ ├── apple
│ │ ├──4
├── hand
├── hand
├── hand
├── hand
├── hand
├── hand
│ ├── position.npy → (1357, 16)
│ ├── action.npy → (1357, 16)
│ └── time.npy → (1357,)
│
├── arm
│ ├── position.npy → (1798, 6)
│ ├── action.npy → (1798, 4, 4)
│ ├── velocity.npy → (1798, 6)
│ ├── torque.npy → (1798, 6)
│ └── time.npy → (1798,)
│
├── Object
│ └── object_6d_pose.npz
│ └── frame_0 ... frame_406
│
├── C2R.npy
│ └── 4×4 coordinate transform
│
└── timestamps
├── frame_id → 1 ... 407
└── timestamp → 407 samples
```

1. Replay the action in viser

```

python -m src.replicate.viser_viewer --hand human --object_name apple --scene 4

```

1. Replay the action in genesis

```

python -m src.replicate.genesis_viewer --hand allegro_v5 --object_name apple --scene 4

```

2. Make contact tensor

```

python -m src.grasping.make_contact_tensor --hand allegro_v5 --object_name apple --scene 4

```

3. Optimize contact tensors

```

python -m src.grasping.contact_optimization --hand allegro_v5 --object_name apple --scene 4

```

4. Make Grasp candidate

```

python -m src.grasping.make_grasp_candidate --hand allegro_v5 --object_name apple --scene 4

```

5. Inspect the dataset

```

python -m src.inspect.dataset --hand HAND --object_name OBJECT_NAME --scene SCENE [--only_hand] [--only_processed] [--only_object] [--only_camera] [--only_timestamps]
example:
python -m src.inspect.dataset --hand allegro_v5 --object_name apple --scene 4 --only_processed

```

6. Train the simple PPO algorithm

```
python -m genesis_env.parallel_envs.train --without_voc --hand allegro_v5 --object_name apple --scene 4 --num_iterations 50 > "logs/output_ppo_2026-09-15.txt"
```

7. Evaluate the simple PPO algorithm

```
python -m genesis_env.parallel_envs.evaluate \
    --checkpoint logs/checkpoints/best_ppo_2026_09_15_11_08.pt \
    --hand allegro_v5 \
    --object_name apple \
    --scene 4
```

8. Train the simple PPO with VOC algorithm

```
python -m genesis_env.parallel_envs.train --hand allegro_v5 --object_name apple --scene 4  --num_iterations 50 > "logs/output_ppo_voc_2026-09-15.txt"
```

9. Evaluate the simple PPO with VOC algorithm

```
python -m genesis_env.parallel_envs.voc.evaluate \
    --hand allegro_v5 \
    --object_name apple \
    --scene 4 \
    --checkpoint logs/checkpoints/best_ppo_voc_2026_09_15_11_03.pt
```

10. Analyze and produce the graph for both algorithms

```
python -m genesis_env.parallel_envs.analyze_ppo_logs \
    --voc "logs/output_ppo_voc_2026_09_15.txt" \
    --baseline "logs/output_ppo_2026_09_15.txt"
```
