"""Export the fixed AIRL reward from a trusted training checkpoint."""

import argparse
from pathlib import Path

from mjlab.rl.frozen_airl import export_frozen_airl


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-file", type=Path, required=True)
  parser.add_argument("--output-file", type=Path, required=True)
  args = parser.parse_args()
  print(
    f"Frozen AIRL reward: {export_frozen_airl(args.checkpoint_file, args.output_file)}"
  )


if __name__ == "__main__":
  main()
