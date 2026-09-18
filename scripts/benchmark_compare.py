"""Benchmark sequential and CUDA CART across dataset sizes and class counts."""

import argparse
import re
import struct
import subprocess
import tempfile
import time
from pathlib import Path

from generate_data import make_samples, write_test_file, write_training_file


PREDICTION = re.compile(r"Object\[(\d+)\]:\s*(-?\d+);")
SEQUENTIAL_TREE = re.compile(r"Creation binary tree time:\s+([0-9.eE+-]+)")
SEQUENTIAL_CLASSIFY = re.compile(r"Time of receiving classes:\s+([0-9.eE+-]+)")
CUDA_TREE = re.compile(r"Tree creation time \(CUDA Parallelized\):\s+([0-9.eE+-]+)")


def read_predictions(path):
    return [int(label) for _, label in PREDICTION.findall(path.read_text())]


def run_program(command, result_path, parser):
    result_path.write_text("")
    started = time.perf_counter()
    completed = subprocess.run(
        command + [str(result_path)], capture_output=True, text=True
    )
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed with exit code {completed.returncode}:\n"
            f"{completed.stdout}{completed.stderr}"
        )
    return parser(completed.stdout), wall_seconds, read_predictions(result_path)


def parse_sequential(output):
    tree = float(SEQUENTIAL_TREE.search(output).group(1))
    classify = float(SEQUENTIAL_CLASSIFY.search(output).group(1))
    return {"tree": tree, "classify": classify, "total": tree + classify}


def parse_cuda(output):
    tree = float(CUDA_TREE.search(output).group(1))
    return {"tree": tree, "total": tree}


def mean(values):
    return sum(values) / len(values)


def benchmark_scenario(
    train_size,
    test_size,
    feature_count,
    classes,
    runs,
    seed,
    sequential_executable,
    cuda_executable,
    directory,
):
    rng = __import__("random").Random(seed)
    train_samples, train_labels = make_samples(
        train_size, feature_count, classes, rng
    )
    test_samples, expected = make_samples(test_size, feature_count, classes, rng)
    train_file = directory / f"train-{train_size}-{classes}.bin"
    test_file = directory / f"test-{train_size}-{classes}.bin"
    write_training_file(train_file, train_samples, train_labels)
    write_test_file(test_file, test_samples)

    sequential_times = {"tree": [], "classify": [], "total": [], "wall": []}
    cuda_times = {"tree": [], "total": [], "wall": []}
    sequential_predictions = []
    cuda_predictions = []

    base_args = [str(train_size), str(feature_count), str(test_size)]
    sequential_command = [
        str(sequential_executable),
        *base_args,
        str(train_file),
        str(test_file),
    ]
    cuda_command = [
        str(cuda_executable),
        *base_args,
        str(train_file),
        str(test_file),
    ]

    for run in range(runs):
        sequential, sequential_wall, sequential_predictions = run_program(
            sequential_command, directory / f"seq-{train_size}-{classes}-{run}.txt", parse_sequential
        )
        cuda, cuda_wall, cuda_predictions = run_program(
            cuda_command, directory / f"cuda-{train_size}-{classes}-{run}.txt", parse_cuda
        )
        for key in sequential_times:
            sequential_times[key].append(
                sequential_wall if key == "wall" else sequential[key]
            )
        for key in cuda_times:
            cuda_times[key].append(cuda_wall if key == "wall" else cuda[key])

    matches = sum(
        left == right for left, right in zip(sequential_predictions, cuda_predictions)
    )
    sequential_accuracy = sum(
        prediction == label
        for prediction, label in zip(sequential_predictions, expected)
    )
    cuda_accuracy = sum(
        prediction == label for prediction, label in zip(cuda_predictions, expected)
    )
    return {
        "train": train_size,
        "test": test_size,
        "classes": classes,
        "seq_tree_ms": mean(sequential_times["tree"]) * 1000,
        "cuda_tree_ms": mean(cuda_times["tree"]) * 1000,
        "tree_speedup": mean(sequential_times["tree"])
        / mean(cuda_times["tree"]),
        "seq_wall_ms": mean(sequential_times["wall"]) * 1000,
        "cuda_wall_ms": mean(cuda_times["wall"]) * 1000,
        "wall_speedup": mean(sequential_times["wall"])
        / mean(cuda_times["wall"]),
        "matches": matches,
        "seq_accuracy": sequential_accuracy / test_size,
        "cuda_accuracy": cuda_accuracy / test_size,
    }


def print_table(rows, test_size):
    columns = [
        ("Train", "train"),
        ("Cls", "classes"),
        ("Seq tree ms", "seq_tree_ms"),
        ("CUDA tree ms", "cuda_tree_ms"),
        ("Tree x", "tree_speedup"),
        ("Seq wall ms", "seq_wall_ms"),
        ("CUDA wall ms", "cuda_wall_ms"),
        ("Wall x", "wall_speedup"),
        ("Match", "matches"),
        ("Seq acc", "seq_accuracy"),
        ("CUDA acc", "cuda_accuracy"),
    ]
    widths = [
        max(len(title), max(len(format_value(row[key], key, test_size)) for row in rows))
        for title, key in columns
    ]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header = "|" + "|".join(f" {title:<{width}} " for (title, _), width in zip(columns, widths)) + "|"
    print(separator)
    print(header)
    print(separator)
    for row in rows:
        print(
            "|"
            + "|".join(
                f" {format_value(row[key], key, test_size):>{width}} "
                for (_, key), width in zip(columns, widths)
            )
            + "|"
        )
    print(separator)


def format_value(value, key, test_size):
    if key in {"seq_tree_ms", "cuda_tree_ms", "seq_wall_ms", "cuda_wall_ms"}:
        return f"{value:.2f}"
    if key in {"tree_speedup", "wall_speedup"}:
        return f"{value:.2f}x"
    if key in {"seq_accuracy", "cuda_accuracy"}:
        return f"{value:.1%}"
    if key == "matches":
        return f"{value}/{test_size}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sizes", default="500,1000,2000")
    parser.add_argument("--class-counts", default="2,3,5")
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=Path, default=Path("build/cart_cuda"))
    parser.add_argument("--sequential", type=Path, default=Path("build/cart_sequential"))
    args = parser.parse_args()

    train_sizes = [int(value) for value in args.train_sizes.split(",")]
    class_counts = [int(value) for value in args.class_counts.split(",")]
    if args.runs <= 0 or args.test_size <= 0 or args.features < 2:
        parser.error("runs and test-size must be positive; features must be at least 2")
    if any(size <= 0 for size in train_sizes):
        parser.error("train sizes must be positive")
    if any(classes < 2 or classes > 32 for classes in class_counts):
        parser.error("class counts must be between 2 and 32")

    sequential_executable = args.sequential.resolve()
    cuda_executable = args.cuda.resolve()
    if not sequential_executable.exists() or not cuda_executable.exists():
        parser.error("compile cart_sequential and cart_cuda_fixed before benchmarking")

    rows = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        for train_size in train_sizes:
            for classes in class_counts:
                rows.append(
                    benchmark_scenario(
                        train_size,
                        args.test_size,
                        args.features,
                        classes,
                        args.runs,
                        args.seed,
                        sequential_executable,
                        cuda_executable,
                        directory,
                    )
                )

    print("CART benchmark comparison")
    print(f"runs per row: {args.runs} | test samples per row: {args.test_size} | features: {args.features}")
    print_table(rows, args.test_size)
    print()
    print("Interpretation:")
    print("- Tree x compares the internally reported tree-construction times.")
    print("- Wall x compares complete process time, including CUDA startup and file I/O.")
    print("- Match is exact prediction agreement between sequential and CUDA.")
    print("- Accuracy compares predictions with the synthetic generator's known rule.")
    return 0 if all(row["matches"] == args.test_size for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())