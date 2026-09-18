"""Compare Object predictions written by the sequential and CUDA CART programs."""

import argparse
import re
from pathlib import Path


PREDICTION = re.compile(r"Object\[(\d+)\]:\s*(-?\d+);")


def read_predictions(path):
    return [int(label) for _, label in PREDICTION.findall(path.read_text())]


def main():
    parser = argparse.ArgumentParser(description="Compare CART prediction result files.")
    parser.add_argument("sequential", type=Path)
    parser.add_argument("cuda", type=Path)
    args = parser.parse_args()

    sequential = read_predictions(args.sequential)
    cuda = read_predictions(args.cuda)
    print(f"sequential predictions: {len(sequential)}")
    print(f"cuda predictions:       {len(cuda)}")

    if len(sequential) != len(cuda):
        print("FAIL: prediction counts differ")
        return 1

    differences = [
        index + 1
        for index, (expected, actual) in enumerate(zip(sequential, cuda))
        if expected != actual
    ]
    matches = len(sequential) - len(differences)
    print(f"matching predictions:    {matches}/{len(sequential)}")
    print(f"different predictions:   {len(differences)}")

    if differences:
        print(f"first differing objects: {differences[:10]}")
        print("FAIL")
        return 1

    print("PASS: CUDA and sequential predictions match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())