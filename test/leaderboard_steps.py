import os
import argparse
import numpy as np

DEFAULT_METHODS = ["lars", "demo_static", "brs", "dpbrs", "rnn", "q"]

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """CAUSAL moving average (no future leakage)."""
    if window <= 1:
        return x.copy()
    x = x.astype(np.float32)
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[window:] - c[:-window]) / window  # length n-window+1
    return np.concatenate([np.full(window-1, ma[0], dtype=np.float32), ma])

def rewards_to_steps(rewards: np.ndarray, max_steps: int) -> np.ndarray:
    # MountainCar: reward is ~ -1 per step, so return ~ -steps
    steps = -rewards.astype(np.float32)
    return np.clip(steps, 1, max_steps)

def pick_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def load_true_reward_file(models_dir: str, method: str, version: str):
    if method == "q":
        return pick_existing([
            os.path.join(models_dir, "q", f"q_rewards_{version}.npy"),
            os.path.join(models_dir, "q", f"q_true_{version}.npy"),
        ])
    return pick_existing([
        os.path.join(models_dir, method, f"{method}_true_{version}.npy"),
        os.path.join(models_dir, method, f"{method}_rewards_{version}.npy"),  # fallback if you ever named it this
    ])

def first_hit_episode(series: np.ndarray, threshold: float, sustain: int = 1):
    """
    Return the first index i where series[i:i+sustain] are all <= threshold.
    If never hits, return None.
    """
    n = len(series)
    if sustain <= 1:
        idx = np.where(series <= threshold)[0]
        return int(idx[0]) if len(idx) else None

    for i in range(0, n - sustain + 1):
        if np.all(series[i:i+sustain] <= threshold):
            return i
    return None

def auc_score(steps_smooth: np.ndarray, upto: int = None):
    """
    Simple learning score: area under steps curve (lower is better).
    """
    if upto is None or upto <= 0 or upto > len(steps_smooth):
        upto = len(steps_smooth)
    return float(np.nanmean(steps_smooth[:upto]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=str, default="v1")
    ap.add_argument("--models_dir", type=str, default="models")
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--thresholds", type=str, default="200,150,120",
                    help="comma-separated thresholds in steps, e.g. 200,150,120")
    ap.add_argument("--sustain", type=int, default=5,
                    help="require staying under threshold for N consecutive episodes")
    ap.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS),
                    help="comma-separated methods list")
    ap.add_argument("--auc_upto", type=int, default=300,
                    help="compute AUC score up to this episode (lower is better)")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]

    rows = []

    for m in methods:
        path = load_true_reward_file(args.models_dir, m, args.version)
        if path is None:
            print(f"[WARN] Missing rewards for {m} (version={args.version})")
            continue

        rewards = np.load(path).astype(np.float32)
        steps = rewards_to_steps(rewards, args.max_steps)
        steps_smooth = moving_average(steps, args.window)

        hits = {thr: first_hit_episode(steps_smooth, thr, sustain=args.sustain) for thr in thresholds}
        score = auc_score(steps_smooth, upto=args.auc_upto)

        rows.append((m, path, hits, score))

    if not rows:
        raise FileNotFoundError("No reward files found. Check --models_dir and --version.")

    # Rank by: AUC first (stability/speed overall), then earliest hit<=200 as tiebreak
    primary_thr = thresholds[0]
    def sort_key(item):
        _, _, hits, score = item
        hit = hits[primary_thr]
        return (score, hit if hit is not None else 10**9)


    rows.sort(key=sort_key)

    # Print leaderboard
    print("\n=== Convergence Leaderboard (lower episode is better) ===")
    print(f"version={args.version} | window={args.window} | sustain={args.sustain} | max_steps={args.max_steps}")
    print(f"AUC score computed up to episode {args.auc_upto} (lower is better)")
    print("--------------------------------------------------------")

    header = ["Rank", "Method", f"Hit<= {int(primary_thr)}", "AUC"]
    for thr in thresholds[1:]:
        header.insert(-1, f"Hit<= {int(thr)}")

    print(" | ".join(header))
    print("-" * 80)

    for i, (m, path, hits, score) in enumerate(rows, start=1):
        cols = [str(i), m.upper() if m != "demo_static" else "DEMO_STATIC"]

        for thr in thresholds:
            ep = hits[thr]
            cols.append(str(ep) if ep is not None else "NA")

        cols.append(f"{score:.2f}")
        print(" | ".join(cols))

    print("\nFiles used:")
    for m, path, _, _ in rows:
        print(f"  {m}: {path}")

if __name__ == "__main__":
    main()
