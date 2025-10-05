"""
Create a stratified train/val split from a single-label txt file:
each line in input should be: <relative/or/absolute_path> <label>
(label should be 0 or 1)
Outputs train.txt and val.txt with same format.
"""

import argparse
import random
import os
from pathlib import Path

def read_labels(labels_file):
    samples = []
    with open(labels_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label = line.rsplit(" ", 1)
            samples.append((path, int(label)))
    return samples

def write_list(samples, out_file):
    with open(out_file, "w") as f:
        for p, lbl in samples:
            f.write(f"{p} {lbl}\n")

def stratified_split(samples, val_size=0.2, seed=42):
    # Separate by label
    pos = [s for s in samples if s[1] == 1]
    neg = [s for s in samples if s[1] == 0]
    random.Random(seed).shuffle(pos)
    random.Random(seed).shuffle(neg)
    def split_list(lst):
        n_val = int(len(lst) * val_size)
        return lst[n_val:], lst[:n_val]  # train, val
    pos_train, pos_val = split_list(pos)
    neg_train, neg_val = split_list(neg)
    train = pos_train + neg_train
    val = pos_val + neg_val
    random.Random(seed).shuffle(train)
    random.Random(seed).shuffle(val)
    return train, val

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels-file", required=True, help="Input labels txt (path label per line)")
    p.add_argument("--out-train", default="train.txt")
    p.add_argument("--out-val", default="val.txt")
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    samples = read_labels(args.labels_file)
    if not samples:
        raise SystemExit("No samples found in labels file.")
    train, val = stratified_split(samples, val_size=args.val_size, seed=args.seed)
    write_list(train, args.out_train)
    write_list(val, args.out_val)
    print(f"Total samples: {len(samples)}; train: {len(train)}; val: {len(val)}")
    print(f"Wrote {args.out_train} and {args.out_val}")

if __name__ == "__main__":
    main()

# python scripts/split_labels.py --labels-file back_of_car_labels.txt --out-train train.txt --out-val val.txt --val-size 0.2
