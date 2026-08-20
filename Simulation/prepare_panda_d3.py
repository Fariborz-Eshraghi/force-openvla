#!/usr/bin/env python3
"""Create the D3 nominal/boundary raw dataset as a filtered view of D2."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_NAME = "panda_pickplace_d3"
ALLOWED_MODES = {"nominal", "boundary"}
EXPECTED_COUNTS = {
    "train": {"nominal": 630, "boundary": 90},
    "test": {"nominal": 70, "boundary": 10},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def ensure_episode_link(source: Path, destination: Path) -> None:
    relative_source = Path(os.path.relpath(source, destination.parent))
    if destination.is_symlink():
        if destination.readlink() != relative_source:
            raise RuntimeError(
                f"Existing D3 link points somewhere unexpected: {destination}"
            )
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing path: {destination}")
    destination.symlink_to(relative_source, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = source / "manifest.jsonl"
    metadata_path = source / "dataset_metadata.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("D2 manifest or metadata is missing")

    source_manifest = read_jsonl(manifest_path)
    selected = [
        row
        for row in source_manifest
        if row["trajectory_mode"] in ALLOWED_MODES
    ]

    actual_counts: dict[str, Counter[str]] = {
        split: Counter(
            row["trajectory_mode"]
            for row in selected
            if row["split"] == split
        )
        for split in EXPECTED_COUNTS
    }
    for split, expected in EXPECTED_COUNTS.items():
        if dict(actual_counts[split]) != expected:
            raise AssertionError(
                f"Unexpected {split} counts: {dict(actual_counts[split])}; "
                f"expected {expected}"
            )

    output.mkdir(parents=True, exist_ok=True)
    for split in EXPECTED_COUNTS:
        (output / split).mkdir(exist_ok=True)

    selected_names: dict[str, set[str]] = {"train": set(), "test": set()}
    transitions = Counter()
    normalized_manifest = []
    for row in selected:
        split = str(row["split"])
        source_episode = source / row["episode_dir"]
        destination = output / split / source_episode.name
        if not (source_episode / "steps.jsonl").is_file():
            raise FileNotFoundError(source_episode / "steps.jsonl")
        ensure_episode_link(source_episode, destination)
        selected_names[split].add(destination.name)
        transitions[split] += int(row["num_steps"])
        normalized_manifest.append(
            {
                **row,
                "episode_dir": f"{split}/{destination.name}",
                "source_dataset": str(source),
            }
        )

    for split in EXPECTED_COUNTS:
        extras = {
            path.name
            for path in (output / split).glob("episode_*")
            if path.name not in selected_names[split]
        }
        if extras:
            raise RuntimeError(
                f"D3 contains unselected episode paths in {split}: {sorted(extras)[:5]}"
            )

    source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = {
        "dataset_name": DATASET_NAME,
        "dataset_version": "1.0.0",
        "description": (
            "D3: successful nominal and boundary Panda pick-place demonstrations "
            "selected from D2; recovery trajectories are excluded."
        ),
        "source_dataset": str(source),
        "completed_episodes": len(selected),
        "train_fraction": source_metadata.get("train_fraction", 0.9),
        "split_counts": {
            split: {
                "episodes": sum(EXPECTED_COUNTS[split].values()),
                "transitions": transitions[split],
                "trajectory_modes": EXPECTED_COUNTS[split],
            }
            for split in EXPECTED_COUNTS
        },
        "included_trajectory_modes": sorted(ALLOWED_MODES),
        "excluded_trajectory_modes": [
            "recovery_approach",
            "recovery_grasp",
            "recovery_transport",
            "recovery_release",
        ],
        "control_hz": source_metadata.get("policy_hz", 10.0),
        "action_format": source_metadata.get(
            "action_format",
            "[dx, dy, dz, droll, dpitch, dyaw, gripper]",
        ),
        "proprio_format": (
            "[q1, q2, q3, q4, q5, q6, q7, "
            "measured_gripper_open_fraction]"
        ),
        "gripper_convention": "1.0=open, 0.0=closed",
    }
    write_json(output / "dataset_metadata.json", metadata)
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in normalized_manifest),
        encoding="utf-8",
    )
    write_json(
        output / "raw_validation_report.json",
        {
            "valid": True,
            "dataset_name": DATASET_NAME,
            "episodes": len(selected),
            "transitions": sum(transitions.values()),
            "counts": {
                split: dict(actual_counts[split]) for split in EXPECTED_COUNTS
            },
            "recovery_episodes": 0,
        },
    )

    print("D3 raw directory:", output)
    print("Episodes: 800 (700 nominal, 100 boundary, 0 recovery)")
    print(
        "Train/test:",
        sum(EXPECTED_COUNTS["train"].values()),
        "/",
        sum(EXPECTED_COUNTS["test"].values()),
    )
    print("Transitions:", dict(transitions))
    print("PANDA_D3_RAW_READY")


if __name__ == "__main__":
    main()
