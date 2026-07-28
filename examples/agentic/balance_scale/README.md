# Odd-Ball Balance-Scale Agentic Example

This example trains a policy to solve the classic odd-ball balance-scale
puzzle: among *N* visually identical balls, exactly one is heavier or lighter
than the rest. The agent uses a balance-scale tool (`weigh`) to compare two
equal-size disjoint groups and a final-answer action (`submit_answer`) to
identify the odd ball and its weight direction.

The environment is deterministic and self-contained — no network services,
no sandbox, no external dependencies beyond the Python standard library.

## Files

- `game.py` — core engine: ball-set creation, weighing simulation, answer
  verification, and prompt formatting.
- `dataset_generator.py` — generates reproducible JSONL puzzles with seeded
  random odd-ball positions and weight directions.
- `dataset_loader.py` — loads JSONL puzzles and converts them to Areno prompt
  records.
- `reward.py` — reward function: scores full-answer accuracy, identity-only
  accuracy, and tracks mean weighings.
- `run_agent.py` — multi-turn agent entrypoint: loops weigh/submit_answer
  tool calls with weighing budget enforcement.

## Generate Puzzles

```bash
python examples/agentic/balance_scale/dataset_generator.py \
  --output /tmp/areno-balance-scale-puzzles.jsonl \
  --count 2048 \
  --seed 2026 \
  --num-balls 12 \
  --max-weighings 3
```

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/areno-balance-scale-puzzles.jsonl \
  --dataset-loader-fn examples/agentic/balance_scale/dataset_loader.py \
  --reward-fn-path examples/agentic/balance_scale/reward.py \
  --agent-fn examples/agentic/balance_scale/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 256
```

## Input Contract

Each JSONL record:

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Unique record identifier |
| `num_balls` | int | Number of balls (default 12) |
| `odd_ball_index` | int | Index of the odd ball (0-based) |
| `direction` | string | `"heavier"` or `"lighter"` |
| `max_weighings` | int | Maximum allowed weighings (default 3) |

## Output & Metrics

The reward function populates `record.metadata` with:

| Field | Type | Description |
| --- | --- | --- |
| `weighings_used` | int | Number of weigh tool calls made |
| `max_weighings` | int | Budget for this puzzle |
| `full_answer_accuracy` | float | 1.0 if ball + direction both correct |
| `identity_only_accuracy` | float | 1.0 if ball index correct (regardless of direction) |

## Reward Rubric

| Outcome | Reward |
| --- | --- |
| Full answer correct (ball + direction) | 1.0 |
| Identity only correct (ball, wrong direction) | 0.5 |
| Submitted but completely wrong | 0.0 |
| No `submit_answer` call / budget exceeded | -1.0 |

## Limitations

- The weighing budget is a hard limit enforced by `run_agent.py`; once
  exhausted the agent is forced to call `submit_answer`.
- The default 12-ball / 3-weighing configuration is information-theoretically
  tight (24 possibilities vs 27 outcomes); models that have not yet learned
  optimal splitting may need more weighings and will score lower.
- `num_balls` and `max_weighings` are configurable but should satisfy
  `3^max_weighings >= num_balls * 2` for the puzzle to be solvable in
  principle.
