import argparse
import subprocess
import sys
import os

METHODS = ["lars", "demo_static", "brs", "dpbrs", "rnn"]

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", type=str, required=True, help="e.g. v3")
    p.add_argument("--train_script", type=str, default="train_comparison_methods.py")
    p.add_argument("--plot_script", type=str, default="plot_steps_comparison.py")
    p.add_argument("--skip_q", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=1000)
    args = p.parse_args()

    q_demo_path = os.path.join("models", "q", f"mountain_car_{args.version}.pkl")

    if not args.skip_train:
        if not args.skip_q:
            run([sys.executable, args.train_script, "--method", "q_only", "--version", args.version, "--train"])

        for m in METHODS:
            run([
                sys.executable, args.train_script,
                "--method", m,
                "--version", args.version,
                "--train",
                "--q_demo_path", q_demo_path
            ])

    if args.plot:
        run([
            sys.executable, args.plot_script,
            "--version", args.version,
            "--window", str(args.window),
            "--max_steps", str(args.max_steps),
            "--out", os.path.join("models", f"comparison_steps_{args.version}.png"),
        ])

if __name__ == "__main__":
    main()
