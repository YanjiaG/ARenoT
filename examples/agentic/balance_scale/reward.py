"""Reward function for the odd-ball balance-scale tool-call example.

Scoring rubric (issue acceptance criteria):

  | outcome                         | reward |
  | ------------------------------- | ------ |
  | Full answer correct (ball + direction) | 1.0 |
  | Identity only correct (ball, wrong direction) | 0.5 |
  | Submitted but completely wrong       | 0.0 |
  | No submit_answer call / budget exceeded | -1.0 |

The function also records ``weighings_used`` and the three accuracy
metrics (``full_answer_accuracy``, ``identity_only_accuracy``,
``mean_weighings``) in ``record.metadata`` for downstream metric reporting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one completion by extracting weigh and submit_answer tool calls."""

    source = record.source_record
    ball_set = game.BallSet(
        num_balls=source["num_balls"],
        odd_ball_index=source["odd_ball_index"],
        direction=source["direction"],
        max_weighings=source["max_weighings"],
    )

    weighings_used = _count_weighings(record)
    answer = _extract_answer(record)

    # Populate metadata for metric aggregation.
    metadata = getattr(record, "metadata", None)
    if metadata is None:
        metadata = {}
    metadata["weighings_used"] = weighings_used
    metadata["max_weighings"] = ball_set.max_weighings

    if answer is None:
        metadata["full_answer_accuracy"] = 0.0
        metadata["identity_only_accuracy"] = 0.0
        _set_metadata(record, metadata)
        return -1.0

    result = game.check_answer(ball_set, answer["ball_index"], answer["direction"])
    metadata["full_answer_accuracy"] = 1.0 if result["full_correct"] else 0.0
    metadata["identity_only_accuracy"] = 1.0 if result["ball_correct"] else 0.0
    _set_metadata(record, metadata)

    if result["full_correct"]:
        return 1.0
    if result["ball_correct"]:
        return 0.5
    return 0.0


def _count_weighings(record: Any) -> int:
    """Count how many weigh tool calls the agent made."""

    count = 0
    for call in _iter_tool_calls(record):
        if call.get("name") == "weigh":
            count += 1
    return count


def _extract_answer(record: Any) -> dict[str, Any] | None:
    """Extract the submit_answer arguments from tool calls."""

    for call in _iter_tool_calls(record):
        if call.get("name") != "submit_answer":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if isinstance(arguments, dict):
            ball_index = arguments.get("ball_index")
            direction = arguments.get("direction")
            try:
                return {"ball_index": int(ball_index), "direction": str(direction)}
            except (TypeError, ValueError):
                return None
    return None


def _iter_tool_calls(record: Any) -> list[dict[str, Any]]:
    """Return the list of tool call dicts from a reward record."""

    tool_calls = getattr(record, "tool_calls", None)
    if tool_calls is None:
        return []
    return tool_calls if isinstance(tool_calls, list) else []


def _set_metadata(record: Any, metadata: dict[str, Any]) -> None:
    """Best-effort write of metadata back onto the record."""

    if hasattr(record, "metadata"):
        if isinstance(record.metadata, dict):
            record.metadata.update(metadata)
        else:
            record.metadata = metadata
