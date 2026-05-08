import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

METHODS = ["lars", "demo_static", "brs", "dpbrs", "rnn", "q"]

def moving_average(x, window):
    if window <= 1:
        return x
    w = np.ones(window) / window
    return np.convolve(x, w, mode="same")

def rewards_to_steps(rewards, max_steps):
    steps = -np.asarray(rewards, dtype=np.float32)
    return np.clip(steps, 1, max_steps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=str, required=True)
    ap.add_argument("--models_dir", type=str, default="models")
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    paths = {}
    for m in METHODS:
        if m == "q":
            p = os.path.join(args.models_dir, "q", f"q_rewards_{args.version}.npy")
        else:
            p = os.path.join(args.models_dir, m, f"{m}_true_{args.version}.npy")

        if os.path.exists(p):
            paths[m] = p

    if not paths:
        raise FileNotFoundError("No reward .npy files found for this version.")

    plt.figure(figsize=(10, 5))

    for m, pth in paths.items():
        rewards = np.load(pth).astype(np.float32)
        steps = rewards_to_steps(rewards, args.max_steps)
        steps_smooth = moving_average(steps, args.window)

        label = m.upper() if m != "demo_static" else "DEMO_STATIC"
        plt.plot(steps_smooth, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Average Steps to Goal (lower is better)")
    plt.title(f"Average Steps to Goal vs Episode (window={args.window}, version={args.version})")
    plt.ylim(0, args.max_steps)
    plt.legend()
    plt.tight_layout()

    out = args.out or os.path.join(args.models_dir, f"comparison_steps_{args.version}.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved -> {out}")
    for m, pth in paths.items():
        print(f"  {m}: {pth}")

if __name__ == "__main__":
    main()
