"""RLDS/TFDS builder for Panda D3 nominal and boundary demonstrations."""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import tensorflow_datasets as tfds
from PIL import Image


RAW_DIR_ENV = "PANDA_D3_RAW_DIR"
ALLOWED_MODES = {"nominal", "boundary"}


def _as_float32(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _quat_wxyz_to_rpy(quat_wxyz: Any) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat_wxyz]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


class PandaPickplaceD3(tfds.core.GeneratorBasedBuilder):
    """Successful nominal/boundary Panda data with joint proprioception."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": (
            "D3 with 700 nominal and 100 boundary episodes, paired cameras, "
            "7D Cartesian actions, and 8D joint/gripper proprioception."
        )
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Fixed third-person RGB observation.",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Synchronized wrist RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc=(
                                            "EEF [x, y, z, roll, pitch, yaw, "
                                            "measured_gripper_open_fraction]."
                                        ),
                                    ),
                                    "joint_qpos": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Measured Franka arm joint positions q1...q7.",
                                    ),
                                    "gripper_open_fraction": tfds.features.Tensor(
                                        shape=(1,),
                                        dtype=np.float32,
                                        doc="Measured gripper opening: 1=open, 0=closed.",
                                    ),
                                    "gripper_qpos": tfds.features.Tensor(
                                        shape=(2,), dtype=np.float32
                                    ),
                                    "cube_pose": tfds.features.Tensor(
                                        shape=(7,), dtype=np.float32
                                    ),
                                    "target_pose": tfds.features.Tensor(
                                        shape=(7,), dtype=np.float32
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc=(
                                    "[dx, dy, dz, droll, dpitch, dyaw, gripper]; "
                                    "rotation is a world-frame relative rotation vector."
                                ),
                            ),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,), dtype=np.float32
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                            "raw_episode_dir": tfds.features.Text(),
                            "episode_id": tfds.features.Scalar(dtype=np.int64),
                            "success": tfds.features.Scalar(dtype=np.bool_),
                            "random_seed": tfds.features.Scalar(dtype=np.int64),
                            "trajectory_mode": tfds.features.Text(),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        raw_value = os.environ.get(RAW_DIR_ENV, "").strip()
        if not raw_value:
            raise RuntimeError(f"Set {RAW_DIR_ENV} to the D3 raw directory.")
        raw_dir = Path(raw_value).expanduser().resolve()
        for split in ("train", "test"):
            if not (raw_dir / split).is_dir():
                raise FileNotFoundError(raw_dir / split)
        return {
            "train": self._generate_examples(raw_dir / "train"),
            "test": self._generate_examples(raw_dir / "test"),
        }

    def _generate_examples(self, split_dir: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
        steps_paths = sorted(glob.glob(str(split_dir / "episode_*" / "steps.jsonl")))
        if not steps_paths:
            raise RuntimeError(f"No D3 episodes found in {split_dir}")

        for steps_path_str in steps_paths:
            steps_path = Path(steps_path_str)
            episode_dir = steps_path.parent
            metadata = json.loads(
                (episode_dir / "episode_metadata.json").read_text(encoding="utf-8")
            )
            mode = str(metadata["trajectory_mode"])
            if mode not in ALLOWED_MODES:
                raise AssertionError(f"Recovery example reached D3 builder: {mode}")
            if not bool(metadata["success"]):
                raise AssertionError(f"Unsuccessful episode reached D3: {episode_dir}")

            rows = [
                json.loads(line)
                for line in steps_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not rows:
                continue

            episode = []
            for index, row in enumerate(rows):
                wrist_path = episode_dir / row["image_before_action"]["agentic"]
                third_path = episode_dir / row["image_before_action"]["third_person"]
                ee_position = _as_float32(row["end_effector_pose"]["position"])
                ee_rpy = _quat_wxyz_to_rpy(row["end_effector_pose"]["quat_wxyz"])
                gripper_open = np.asarray(
                    [float(row["measured_gripper_open_fraction"])],
                    dtype=np.float32,
                )
                if not 0.0 <= float(gripper_open[0]) <= 1.0:
                    raise AssertionError(f"Invalid gripper opening in {steps_path}")
                state = np.concatenate([ee_position, ee_rpy, gripper_open]).astype(
                    np.float32
                )
                joint_qpos = _as_float32(
                    [row["joint_qpos"][f"joint{joint}"] for joint in range(1, 8)]
                )
                gripper_qpos = _as_float32(
                    [
                        row["gripper_qpos"]["finger_joint1"],
                        row["gripper_qpos"]["finger_joint2"],
                    ]
                )
                if not np.all(np.isfinite(joint_qpos)):
                    raise AssertionError(f"Non-finite joint state in {steps_path}")
                cube_pose = _as_float32(
                    row["cube_pose"]["position"] + row["cube_pose"]["quat_wxyz"]
                )
                target_pose = _as_float32(
                    row["target_corner_pose"]["position"]
                    + row["target_corner_pose"]["quat_wxyz"]
                )
                action = _as_float32(row["action"])
                if action.shape != (7,) or not np.all(np.isfinite(action)):
                    raise AssertionError(f"Invalid action in {steps_path}")
                is_last = index == len(rows) - 1
                episode.append(
                    {
                        "observation": {
                            "image": _load_rgb(third_path),
                            "wrist_image": _load_rgb(wrist_path),
                            "state": state,
                            "joint_qpos": joint_qpos,
                            "gripper_open_fraction": gripper_open,
                            "gripper_qpos": gripper_qpos,
                            "cube_pose": cube_pose,
                            "target_pose": target_pose,
                        },
                        "action": action,
                        "discount": np.float32(1.0),
                        "reward": np.float32(1.0 if is_last else 0.0),
                        "is_first": index == 0,
                        "is_last": is_last,
                        "is_terminal": is_last,
                        "language_instruction": row["language_instruction"],
                        "language_embedding": np.zeros((512,), dtype=np.float32),
                    }
                )

            yield f"{split_dir.name}_{episode_dir.name}", {
                "steps": episode,
                "episode_metadata": {
                    "file_path": str(steps_path),
                    "raw_episode_dir": str(episode_dir),
                    "episode_id": int(metadata["episode_id"]),
                    "success": True,
                    "random_seed": int(metadata["seed"]),
                    "trajectory_mode": mode,
                },
            }
