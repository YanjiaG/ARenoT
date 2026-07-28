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
- `reward.py` — reward function: information-gain-aware continuous scoring
  with repetition/invalid penalties, auto-scales to any number of balls.
- `run_agent.py` — multi-turn agent entrypoint: loops weigh/submit_answer
  tool calls with budget enforcement.
- `verify_ui.py` — Gradio verification UI: configure puzzles and observe the
  model's reasoning trace and final verdict.

## Generate Puzzles

```bash
python examples/agentic/balance_scale/dataset_generator.py \
  --output /tmp/areno-balance-scale-puzzles.jsonl \
  --count 2048 \
  --seed 2026 \
  --num-balls 12
```

When `--max-weighings` is omitted (or 0), it auto-computes as 2× the
information-theoretic minimum: `ceil(log3(num_balls * 2))`.

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /tmp/areno-balance-scale-puzzles.jsonl \
  --dataset-loader-fn examples/agentic/balance_scale/dataset_loader.py \
  --reward-fn-path examples/agentic/balance_scale/reward.py \
  --agent-fn examples/agentic/balance_scale/run_agent.py \
  --algo gspo \
  --batch-size 1 \
  --n-samples 2 \
  --max-new-tokens 64 \
  --world-size 1 \
  --tp-size 1
```

## Input Contract

Each JSONL record:

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Unique record identifier |
| `num_balls` | int | Number of balls (default 12) |
| `odd_ball_index` | int | Index of the odd ball (0-based) |
| `direction` | string | `"heavier"` or `"lighter"` |
| `max_weighings` | int | Soft upper bound on weighings (auto = 2× ceil(log3(num_balls*2))) |

## Output & Metrics

The reward function populates `record.metadata` with:

| Field | Type | Description |
| --- | --- | --- |
| `weighings_used` | int | Total weigh tool calls (valid + repeated + invalid) |
| `valid_weighings` | int | Unique valid weighings |
| `repeated_weighings` | int | Duplicate weighings (same groups) |
| `invalid_weighings` | int | Invalid weighings (bad size/overlap/range) |
| `min_weighings` | int | Information-theoretic minimum: ceil(log3(num_balls*2)) |
| `base_reward` | int | Base reward = min_weighings |
| `full_answer_accuracy` | float | 1.0 if ball + direction both correct |
| `identity_only_accuracy` | float | 1.0 if ball index correct (regardless of direction) |
| `reward_components` | dict | Breakdown: k, t_cost, repeat_cost, invalid_cost |

## Reward Formula

```
R_end = K - T·alpha - P_repeat - P_invalid
```

| Component | Value | Description |
| --- | --- | --- |
| K (answer reward) | `base` / `base/2` / `0` / `-1` | Full correct / identity only / wrong / no submit |
| base | `ceil(log3(num_balls * 2))` | Information-theoretic minimum, auto-scales |
| T (weighing cost) | `(valid + repeated) * alpha` | Each weighing costs `alpha` (default 0.15) |
| P_repeat | `repeated * 0.3` | Penalty for identical repeated weighings |
| P_invalid | `invalid * 0.2` | Penalty for malformed weighings |

### Example rewards (12 balls, base=3)

| Scenario | Weighings | Reward |
| --- | --- | --- |
| 0 weighings, full correct | 0 | 3.0 (lucky guess) |
| 3 weighings, full correct | 3 | 3.0 - 0.45 = 2.55 |
| 3 weighings, identity only | 3 | 1.5 - 0.45 = 1.05 |
| 3 weighings, wrong | 3 | 0.0 - 0.45 = -0.45 |
| No submit | 3 | -1.0 |

## Verification UI

After training, launch the Gradio UI to interactively test the model:

```bash
python examples/agentic/balance_scale/verify_ui.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model policy
```

Or without an LLM (random agent for demo):

```bash
python examples/agentic/balance_scale/verify_ui.py --agent-mode random
```

The UI lets you:
- Set the number of balls (2–200)
- Set the odd ball index and direction (or randomize)
- Watch the model's multi-turn weighing trace
- See the final verdict: correct/wrong, weighings used, efficiency vs optimal

In Colab, the UI appears inline with `share=True`.

## Limitations

- The weighing budget is a soft upper bound (auto = 2× theoretical minimum);
  the agent loop enforces it by forcing `submit_answer` when exhausted.
- The reward auto-scales via `ceil(log3(num_balls * 2))`, so larger ball
  counts produce higher base rewards and proportionally higher weighing costs.
- GPU training requires NVIDIA GPU with sufficient VRAM; T4 (15GB) can run
  rollout + 1 training step with `--batch-size 1 --max-new-tokens 64`.
