"""Generate binary data files consumed by cart_for_cuda_fixed.cu."""

import argparse
import random
import struct
from pathlib import Path


def make_samples(count, feature_count, classes, rng):
	"""Create samples with a deterministic, learnable class boundary."""
	samples = []
	labels = []
	for _ in range(count):
		features = [rng.uniform(-1.0, 1.0) for _ in range(feature_count)]
		score = features[0] + 0.35 * features[1]
		bucket = int((score + 1.35) * classes / 2.7)
		label = min(classes - 1, bucket)
		samples.append(features)
		labels.append(label)
	return samples, labels


def write_training_file(path, samples, labels):
	with path.open("wb") as output:
		for row in samples:
			output.write(struct.pack(f"<{len(row)}d", *row))
		output.write(struct.pack(f"<{len(labels)}i", *labels))


def write_test_file(path, samples):
	with path.open("wb") as output:
		for row in samples:
			output.write(struct.pack(f"<{len(row)}d", *row))


def main():
	parser = argparse.ArgumentParser(
		description="Generate binary train/test files for cart_for_cuda_fixed.cu."
	)
	parser.add_argument("--train-size", type=int, default=1000)
	parser.add_argument("--test-size", type=int, default=200)
	parser.add_argument("--features", type=int, default=4)
	parser.add_argument("--classes", type=int, default=2)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--train-file", type=Path, default=Path("data/train.bin"))
	parser.add_argument("--test-file", type=Path, default=Path("data/test.bin"))
	args = parser.parse_args()

	if args.train_size <= 0 or args.test_size < 0 or args.features < 2:
		parser.error("train-size and features must be positive; features must be at least 2")
	if args.classes < 2 or args.classes > 32:
		parser.error("classes must be between 2 and 32")

	rng = random.Random(args.seed)
	train_samples, labels = make_samples(args.train_size, args.features, args.classes, rng)
	test_samples, _ = make_samples(args.test_size, args.features, args.classes, rng)

	write_training_file(args.train_file, train_samples, labels)
	write_test_file(args.test_file, test_samples)

	print(
		f"Generated {args.train_file} ({args.train_size} samples, "
		f"{args.features} features, {args.classes} classes) and "
		f"{args.test_file} ({args.test_size} samples)."
	)


if __name__ == "__main__":
	main()
