#!/usr/bin/env python3
"""Verify D3 data, joint proprioception, and OpenVLA-OFT action chunks."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="panda_pickplace_d3")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.expanduser().resolve()))
    sys.argv.append("panda_d3")

    from prismatic.vla.constants import (
        ACTION_DIM,
        NUM_ACTIONS_CHUNK,
        PROPRIO_DIM,
    )
    from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
    from prismatic.vla.datasets.rlds.oxe.configs import (
        OXE_DATASET_CONFIGS,
        StateEncoding,
    )
    from prismatic.vla.datasets.rlds.oxe.materialize import (
        make_oxe_dataset_kwargs,
    )
    from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
    from prismatic.vla.datasets.rlds.oxe.transforms import (
        OXE_STANDARDIZATION_TRANSFORMS,
    )
    from prismatic.vla.datasets.rlds.traj_transforms import chunk_act_obs

    assert ACTION_DIM == 7
    assert PROPRIO_DIM == 8
    assert NUM_ACTIONS_CHUNK == 8

    builder = tfds.builder(args.dataset_name, data_dir=str(args.data_root))
    train_count = int(builder.info.splits["train"].num_examples)
    test_count = int(builder.info.splits["test"].num_examples)
    assert (train_count, test_count) == (720, 80), (train_count, test_count)

    config = OXE_DATASET_CONFIGS[args.dataset_name]
    assert config["image_obs_keys"]["primary"] == "image"
    assert config["state_obs_keys"] == [
        "joint_qpos",
        "gripper_open_fraction",
    ]
    assert config["state_encoding"] is StateEncoding.JOINT
    assert OXE_NAMED_MIXTURES[args.dataset_name] == [
        (args.dataset_name, 1.0)
    ]

    transform = OXE_STANDARDIZATION_TRANSFORMS[args.dataset_name]
    transform_source = inspect.getsource(transform)
    assert "recovery" not in transform_source.lower()

    raw_dataset = tfds.load(
        args.dataset_name,
        split="train",
        data_dir=str(args.data_root),
        shuffle_files=False,
    )
    raw_episode = next(iter(raw_dataset.take(1)))
    raw_trajectory = next(iter(raw_episode["steps"].batch(2000)))
    raw_actions = raw_trajectory["action"].numpy()
    raw_joints = raw_trajectory["observation"]["joint_qpos"].numpy()
    raw_gripper = raw_trajectory["observation"][
        "gripper_open_fraction"
    ].numpy()
    expected_proprio = np.concatenate([raw_joints, raw_gripper], axis=1)
    assert raw_actions.shape[1] == 7
    assert expected_proprio.shape[1] == 8
    assert np.all(np.isfinite(expected_proprio))
    assert np.all((raw_gripper >= 0.0) & (raw_gripper <= 1.0))

    kwargs = make_oxe_dataset_kwargs(
        args.dataset_name,
        args.data_root,
        load_camera_views=("primary", "wrist"),
        load_proprio=True,
        load_language=True,
    )
    encoded, stats = make_dataset_from_rlds(
        **kwargs,
        train=True,
        shuffle=False,
        num_parallel_reads=1,
        num_parallel_calls=1,
    )
    assert int(stats["num_trajectories"]) == 720
    assert int(stats["num_transitions"]) > 0
    assert np.asarray(stats["proprio"]["std"]).shape == (8,)
    assert np.all(np.asarray(stats["proprio"]["std"]) > 0.0)

    encoded_episode = next(iter(encoded.take(1)))
    assert encoded_episode["observation"]["proprio"].shape[-1] == 8
    chunked = chunk_act_obs(
        encoded_episode,
        window_size=1,
        future_action_window_size=NUM_ACTIONS_CHUNK - 1,
    )
    assert chunked["action"].shape[1:] == (8, 7)
    assert chunked["observation"]["proprio"].shape[1:] == (1, 8)

    print("Dataset:", builder.info.full_name)
    print(f"Official splits: train={train_count}, test={test_count}")
    print("Proprio: [q1, q2, q3, q4, q5, q6, q7, gripper_open_fraction]")
    print("Proprio q01:", np.asarray(stats["proprio"]["q01"]).tolist())
    print("Proprio q99:", np.asarray(stats["proprio"]["q99"]).tolist())
    print("OFT action chunk shape:", tuple(chunked["action"].shape[1:]))
    print("OPENVLA_OFT_D3_VERIFIED")


if __name__ == "__main__":
    main()
