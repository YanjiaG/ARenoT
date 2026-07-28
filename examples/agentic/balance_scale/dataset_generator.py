"""Generate odd-ball balance-scale puzzles for the agentic example."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

DEFAULT_COUNT = 128
DEFAULT_SEED = 2026
DEFAULT_NUM_BALLS = 12
DEFAULT_MAX_WEIGHINGS = 3


def generate_records(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    num_balls: int = DEFAULT_NUM_BALLS,
    max_weighings: int = DEFAULT_MAX_WEIGHINGS,
) -> list[dict]:
    """Generate reproducible odd-ball puzzle records.

    Each record contains ``num_balls``, ``odd_ball_index``, ``direction``
    (``"heavier"`` or ``"lighter"``), and ``max_weighings``. A single seed
    produces the same sequence every time.
    """

    rng = random.Random(seed)
    records: list[dict] = []
    seen: set[tuple[int, str]] = set()
    attempts = 0
    max_unique = num_balls * 2  # each ball can be heavier or lighter
    target = min(count, max_unique)

    while len(records) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError(
                f"could not generate {count} unique puzzles with {num_balls} balls"
            )

        odd_ball_index = rng.randint(0, num_balls - 1)
        direction = rng.choice(game.DIRECTIONS)
        key = (odd_ball_index, direction)

        if len(records) < target:
            if key in seen:
                continue
            seen.add(key)

        records.append(
            {
                "id": f"generated-{len(records):05d}",
                "num_balls": num_balls,
                "odd_ball_index": odd_ball_index,
                "direction": direction,
                "max_weighings": max_weighings,
            }
        )
    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL puzzles for the Areno odd-ball balance-scale agentic example."
    )
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of puzzles to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--num-balls", type=int, default=DEFAULT_NUM_BALLS, help="Number of balls per puzzle.")
    parser.add_argument(
        "--max-weighings", type=int, default=DEFAULT_MAX_WEIGHINGS, help="Maximum weighings allowed."
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.num_balls < 2:
        raise ValueError("--num-balls must be at least 2")
    if args.max_weighings < 1:
        raise ValueError("--max-weighings must be at least 1")

    records = generate_records(
        args.count,
        seed=args.seed,
        num_balls=args.num_balls,
        max_weighings=args.max_weighings,
    )
    if args.output == "-":
        write_jsonl(records, sys.stdout)
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            write_jsonl(records, handle)


if __name__ == "__main__":
    main()
