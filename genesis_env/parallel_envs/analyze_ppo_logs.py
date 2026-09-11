#!/usr/bin/env python3
"""
Convert PPO console output logs to .npy and generate comparison graphs/metrics.

Expected log lines include:
Episode 0000 | Mean 340.6561 | Best 349.9663 (env 107) |
Actor 0.33220 | Critic 2.72368 | Ratio 0.0000 | VOC mean 1.0000

For non-VOC logs, VOC mean can be absent.

Usage:
    python analyze_ppo_logs.py --voc voc.txt --baseline baseline.txt

It creates:
    ppo_analysis/
        voc_metrics.npy
        baseline_metrics.npy
        comparison_metrics.npy
        metrics_summary.txt
        *.png
"""

import argparse
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

PATTERN = re.compile(
    r"Episode\s+(\d+)\s*\|\s*"
    r"Mean\s+([-+0-9.eE]+)\s*\|\s*"
    r"Best\s+([-+0-9.eE]+)\s*\(env\s+(\d+)\)\s*\|\s*"
    r"Actor\s+([-+0-9.eE]+)\s*\|\s*"
    r"Critic\s+([-+0-9.eE]+)\s*\|\s*"
    r"Ratio\s+([-+0-9.eE]+)"
    r"(?:\s*\|\s*VOC mean\s+([-+0-9.eE]+))?"
)

DTYPE = [
    ("episode", np.int32),
    ("mean_reward", np.float64),
    ("best_reward", np.float64),
    ("best_env", np.int32),
    ("actor_loss", np.float64),
    ("critic_loss", np.float64),
    ("ratio", np.float64),
    ("voc_mean", np.float64),
]


def parse_log(path):
    text = Path(path).read_text(errors="replace")
    rows = []

    for line in text.splitlines():
        m = PATTERN.search(line)
        if not m:
            continue

        episode, mean_r, best_r, best_env, actor, critic, ratio, voc = m.groups()
        rows.append((
            int(episode),
            float(mean_r),
            float(best_r),
            int(best_env),
            float(actor),
            float(critic),
            float(ratio),
            np.nan if voc is None else float(voc),
        ))

    if not rows:
        raise ValueError(f"No Episode lines found in: {path}")

    rows.sort(key=lambda x: x[0])
    data = np.array(rows, dtype=DTYPE)

    # Keep the best-so-far curve available even if the printed "Best"
    # is only the best environment in that episode.
    from numpy.lib import recfunctions as rfn
    data = rfn.append_fields(
        data,
        "best_so_far",
        np.maximum.accumulate(data["best_reward"]),
        usemask=False,
    )
    return data


def save_npy(data, path):
    np.save(path, data, allow_pickle=False)


def moving_average(x, window):
    if len(x) < window:
        return np.array([])
    return np.convolve(x, np.ones(window) / window, mode="valid")


def summarize(data, name):
    mean = data["mean_reward"]
    best = data["best_reward"]

    return {
        "name": name,
        "episodes": len(data),
        "mean_reward_final": float(mean[-1]),
        "mean_reward_max": float(mean.max()),
        "mean_reward_mean": float(mean.mean()),
        "best_reward_max": float(best.max()),
        "best_so_far_final": float(data["best_so_far"][-1]),
        "actor_loss_final": float(data["actor_loss"][-1]),
        "critic_loss_final": float(data["critic_loss"][-1]),
        "ratio_final": float(data["ratio"][-1]),
    }


def print_comparison(voc, base):
    print("\n========== COMPARISON ==========")

    rows = [
        ("Episodes", len(voc), len(base)),
        ("Final mean reward", voc["mean_reward"][-1], base["mean_reward"][-1]),
        ("Maximum mean reward", voc["mean_reward"].max(), base["mean_reward"].max()),
        ("Maximum episode best", voc["best_reward"].max(), base["best_reward"].max()),
        ("Final best-so-far", voc["best_so_far"][-1], base["best_so_far"][-1]),
        ("Final actor loss", voc["actor_loss"][-1], base["actor_loss"][-1]),
        ("Final critic loss", voc["critic_loss"][-1], base["critic_loss"][-1]),
    ]

    for label, v, b in rows:
        if isinstance(v, (int, np.integer)):
            print(f"{label:24s}: VOC={v} | Baseline={b}")
        else:
            delta = v - b
            pct = (delta / abs(b) * 100.0) if b != 0 else np.nan
            print(
                f"{label:24s}: VOC={v:.6f} | "
                f"Baseline={b:.6f} | Δ={delta:+.6f} ({pct:+.2f}%)"
            )

    voc_curve = voc["mean_reward"]
    base_curve = base["mean_reward"]
    n = min(len(voc_curve), len(base_curve))

    if n:
        voc_avg = voc_curve[:n].mean()
        base_avg = base_curve[:n].mean()
        print(f"\nMean reward over common episodes ({n}):")
        print(f"  VOC     : {voc_avg:.6f}")
        print(f"  Baseline: {base_avg:.6f}")
        if base_avg != 0:
            print(f"  Relative: {(voc_avg-base_avg)/abs(base_avg)*100:+.2f}%")


def plot_training(voc, base, outdir, ma_window=5):
    outdir = Path(outdir)

    # 1. Mean reward
    plt.figure(figsize=(10, 6))
    plt.plot(base["episode"], base["mean_reward"], marker="o", label="PPO")
    plt.plot(voc["episode"], voc["mean_reward"], marker="o", label="PPO + VOC")
    if len(base) >= ma_window:
        x = base["episode"][ma_window-1:]
        plt.plot(x, moving_average(base["mean_reward"], ma_window),
                 linewidth=2, label=f"PPO {ma_window}-ep MA")
    if len(voc) >= ma_window:
        x = voc["episode"][ma_window-1:]
        plt.plot(x, moving_average(voc["mean_reward"], ma_window),
                 linewidth=2, label=f"VOC {ma_window}-ep MA")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.title("Training Mean Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "01_mean_reward.png", dpi=200)
    plt.close()

    # 2. Best reward
    plt.figure(figsize=(10, 6))
    plt.plot(base["episode"], base["best_reward"], marker="o", label="PPO")
    plt.plot(voc["episode"], voc["best_reward"], marker="o", label="PPO + VOC")
    plt.plot(base["episode"], base["best_so_far"], "--", label="PPO best-so-far")
    plt.plot(voc["episode"], voc["best_so_far"], "--", label="VOC best-so-far")
    plt.xlabel("Episode")
    plt.ylabel("Best Reward")
    plt.title("Best Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "02_best_reward.png", dpi=200)
    plt.close()

    # 3. Actor loss
    plt.figure(figsize=(10, 6))
    plt.plot(base["episode"], base["actor_loss"], marker="o", label="PPO Actor")
    plt.plot(voc["episode"], voc["actor_loss"], marker="o", label="VOC Actor")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Actor Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "03_actor_loss.png", dpi=200)
    plt.close()

    # 3. Actor/Critic loss
    plt.figure(figsize=(10, 6))
    plt.plot(base["episode"], base["critic_loss"], marker="o", label="PPO Critic")
    plt.plot(voc["episode"], voc["critic_loss"], marker="o", label="VOC Critic")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Critic Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "03_critic_loss.png", dpi=200)
    plt.close()

    # 4. Ratio
    plt.figure(figsize=(10, 6))
    plt.plot(base["episode"], base["ratio"], marker="o", label="PPO")
    plt.plot(voc["episode"], voc["ratio"], marker="o", label="PPO + VOC")
    plt.xlabel("Episode")
    plt.ylabel("Policy Ratio")
    plt.title("PPO Policy Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "04_policy_ratio.png", dpi=200)
    plt.close()

    # 5. VOC strength
    if not np.all(np.isnan(voc["voc_mean"])):
        plt.figure(figsize=(10, 6))
        plt.plot(voc["episode"], voc["voc_mean"], marker="o")
        plt.xlabel("Episode")
        plt.ylabel("VOC Mean Strength")
        plt.title("VOC Strength During Training")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "05_voc_strength.png", dpi=200)
        plt.close()

    # 6. Reward improvement / difference
    n = min(len(voc), len(base))
    diff = voc["mean_reward"][:n] - base["mean_reward"][:n]
    pct = np.divide(
        diff,
        np.abs(base["mean_reward"][:n]),
        out=np.full_like(diff, np.nan),
        where=base["mean_reward"][:n] != 0,
    ) * 100

    plt.figure(figsize=(10, 6))
    plt.axhline(0, linewidth=1)
    plt.plot(voc["episode"][:n], diff, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("VOC − PPO Mean Reward")
    plt.title("Reward Difference: PPO + VOC vs PPO")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "06_reward_difference.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.axhline(0, linewidth=1)
    plt.plot(voc["episode"][:n], pct, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Improvement (%)")
    plt.title("Relative Mean-Reward Improvement")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "07_reward_improvement_percent.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc", required=False, help="VOC training output text file")
    parser.add_argument("--baseline", required=False, help="Normal PPO output text file")
    parser.add_argument("--indir", required=False, help="Input directory for PPO npy files (if already converted)")
    parser.add_argument("--outdir", default="logs/ppo_analysis")
    parser.add_argument("--ma-window", type=int, default=5)
    args = parser.parse_args()

    if args.indir:
        voc = np.load(Path(args.indir) / "voc_metrics.npy", allow_pickle=False)
        base = np.load(Path(args.indir) / "baseline_metrics.npy", allow_pickle=False)
    else:
        if not args.voc or not args.baseline:
            parser.error("Either --indir or both --voc and --baseline must be provided.")
        voc = parse_log(args.voc)
        base = parse_log(args.baseline)

        save_npy(voc, outdir / "voc_metrics.npy")
        save_npy(base, outdir / "baseline_metrics.npy")


    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save a common comparison array.
    n = min(len(voc), len(base))
    comparison = np.column_stack([
        voc["episode"][:n],
        voc["mean_reward"][:n],
        base["mean_reward"][:n],
        voc["best_reward"][:n],
        base["best_reward"][:n],
        voc["actor_loss"][:n],
        base["actor_loss"][:n],
        voc["critic_loss"][:n],
        base["critic_loss"][:n],
        voc["ratio"][:n],
        base["ratio"][:n],
    ])
    np.save(outdir / "comparison_metrics.npy", comparison)

    plot_training(voc, base, outdir, args.ma_window)

    sv = summarize(voc, "PPO + VOC")
    sb = summarize(base, "PPO")

    with open(outdir / "metrics_summary.txt", "w") as f:
        f.write("PPO vs PPO + VOC\n")
        f.write("================\n\n")
        for s in (sv, sb):
            f.write(f"{s['name']}\n")
            for k, v in s.items():
                if k != "name":
                    f.write(f"  {k}: {v}\n")
            f.write("\n")

    print_comparison(voc, base)

    print("\n========== FILES SAVED ==========")
    for p in sorted(outdir.iterdir()):
        print(p)


if __name__ == "__main__":
    main()
