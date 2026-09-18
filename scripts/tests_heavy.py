"""Run longer CUDA-versus-CPU CART benchmarks."""

import argparse
import subprocess
from pathlib import Path


def gpu_summary():
	command = [
		"nvidia-smi",
		"--query-gpu=name,driver_version,memory.total,compute_cap",
		"--format=csv,noheader",
	]
	try:
		result = subprocess.run(command, capture_output=True, text=True, check=True)
	except (FileNotFoundError, subprocess.CalledProcessError):
		return "GPU information unavailable"
	return result.stdout.strip() or "GPU information unavailable"


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--train-sizes", default="2000,4000,8000")
	parser.add_argument("--class-counts", default="2,5,10")
	parser.add_argument("--test-size", type=int, default=500)
	parser.add_argument("--features", type=int, default=4)
	parser.add_argument("--runs", type=int, default=2)
	args = parser.parse_args()

	benchmark = Path(__file__).with_name("benchmark_compare.py")
	command = [
		"python3",
		str(benchmark),
		"--train-sizes",
		args.train_sizes,
		"--class-counts",
		args.class_counts,
		"--test-size",
		str(args.test_size),
		"--features",
		str(args.features),
		"--runs",
		str(args.runs),
	]

	print("GPU configuration", flush=True)
	print(gpu_summary(), flush=True)
	print(flush=True)
	print("Heavy benchmark command", flush=True)
	print(" ".join(command), flush=True)
	print(flush=True)
	subprocess.run(command, check=True)


if __name__ == "__main__":
	main()
