#!/usr/bin/env python3
"""Register Panda D3 joint-proprio data in an OpenVLA-OFT checkout."""

from __future__ import annotations

import argparse
import py_compile
import re
import shutil
from pathlib import Path


DATASET_NAME = "panda_pickplace_d3"

CONFIG_BLOCK = '''
    # BEGIN PANDA_D3_CONFIG
    "panda_pickplace_d3": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_qpos", "gripper_open_fraction"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    # END PANDA_D3_CONFIG
'''

TRANSFORM_BLOCK = '''

# BEGIN PANDA_D3_TRANSFORM
def panda_pickplace_d3_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    """Keep the 7D Cartesian action and measured 8D joint proprioception."""
    action = tf.cast(trajectory["action"], tf.float32)
    trajectory["action"] = tf.concat(
        [action[:, :6], tf.clip_by_value(action[:, -1:], 0.0, 1.0)],
        axis=1,
    )
    return trajectory
# END PANDA_D3_TRANSFORM
'''

CONSTANTS_BLOCK = '''

# BEGIN PANDA_D3_CONSTANTS
PANDA_D3_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}
# END PANDA_D3_CONSTANTS
'''

MIXTURE_LINE = '    "panda_pickplace_d3": [("panda_pickplace_d3", 1.0)],\n'


def backup_once(path: Path) -> None:
    backup = path.with_name(path.name + ".pre_panda_d3_backup")
    if not backup.exists():
        shutil.copy2(path, backup)
        print("Backup:", backup)


def upsert_marked_block(
    path: Path,
    begin: str,
    end: str,
    block: str,
    anchor: str,
) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"\n[ \t]*{re.escape(begin)}.*?{re.escape(end)}\n",
        re.DOTALL,
    )
    if pattern.search(text):
        text = pattern.sub("\n" + block.lstrip("\n"), text, count=1)
    else:
        if anchor not in text:
            raise RuntimeError(f"Could not locate anchor {anchor!r} in {path}")
        text = text.replace(anchor, anchor + block, 1)
    path.write_text(text, encoding="utf-8")


def patch_config(path: Path) -> None:
    upsert_marked_block(
        path,
        "# BEGIN PANDA_D3_CONFIG",
        "# END PANDA_D3_CONFIG",
        CONFIG_BLOCK,
        "OXE_DATASET_CONFIGS = {\n",
    )


def patch_transform(path: Path) -> None:
    upsert_marked_block(
        path,
        "# BEGIN PANDA_D3_TRANSFORM",
        "# END PANDA_D3_TRANSFORM",
        TRANSFORM_BLOCK,
        "\n\n# === Registry ===",
    )
    text = path.read_text(encoding="utf-8")
    registry = (
        '    "panda_pickplace_d3": '
        "panda_pickplace_d3_dataset_transform,\n"
    )
    if registry not in text:
        anchor = "OXE_STANDARDIZATION_TRANSFORMS = {\n"
        if anchor not in text:
            raise RuntimeError(f"Could not locate transform registry in {path}")
        text = text.replace(anchor, anchor + registry, 1)
    path.write_text(text, encoding="utf-8")


def patch_mixture(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    existing = re.compile(
        r'^    "panda_pickplace_d3": .*?\n', re.MULTILINE
    )
    if existing.search(text):
        text = existing.sub(MIXTURE_LINE, text, count=1)
    else:
        anchor = "OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {\n"
        if anchor not in text:
            raise RuntimeError(f"Could not locate mixture registry in {path}")
        text = text.replace(anchor, anchor + MIXTURE_LINE, 1)
    path.write_text(text, encoding="utf-8")


def patch_constants(path: Path) -> None:
    upsert_marked_block(
        path,
        "# BEGIN PANDA_D3_CONSTANTS",
        "# END PANDA_D3_CONSTANTS",
        CONSTANTS_BLOCK,
        "\n\n# Function to detect robot platform from command line arguments",
    )
    text = path.read_text(encoding="utf-8")
    if "# PANDA_D3_PLATFORM_DETECTION" not in text:
        old = '''    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
'''
        new = '''    # PANDA_D3_PLATFORM_DETECTION
    if "panda_pickplace_d3" in cmd_args or "panda_d3" in cmd_args:
        return "PANDA_D3"
    elif "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
'''
        if old not in text:
            raise RuntimeError(f"Could not patch platform detection in {path}")
        text = text.replace(old, new, 1)

    if 'if ROBOT_PLATFORM == "PANDA_D3":' not in text:
        old = '''if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
'''
        new = '''if ROBOT_PLATFORM == "PANDA_D3":
    constants = PANDA_D3_CONSTANTS
elif ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
'''
        if old not in text:
            raise RuntimeError(f"Could not patch platform constants in {path}")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


def patch_official_splits(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "# PANDA_D3_TRAIN_ONLY_STATISTICS" not in text:
        old = '''    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(
            builder, split="all", shuffle=False, num_parallel_reads=num_parallel_reads
        ).traj_map(restructure, num_parallel_calls)
'''
        new = '''    elif dataset_statistics is None:
        # PANDA_D3_TRAIN_ONLY_STATISTICS
        statistics_split = (
            "train"
            if name == "panda_pickplace_d3" and "train" in builder.info.splits
            else "all"
        )
        full_dataset = dl.DLataset.from_rlds(
            builder,
            split=statistics_split,
            shuffle=False,
            num_parallel_reads=num_parallel_reads,
        ).traj_map(restructure, num_parallel_calls)
'''
        if old not in text:
            raise RuntimeError(f"Could not patch statistics split in {path}")
        text = text.replace(old, new, 1)

    if "# PANDA_D3_OFFICIAL_SPLITS" not in text:
        old = '    split = "train" if train else "val"\n'
        new = '''    # PANDA_D3_OFFICIAL_SPLITS
    if name == "panda_pickplace_d3" and "test" in builder.info.splits:
        split = "train" if train else "test"
    else:
        split = "train" if train else "val"
'''
        if old not in text:
            raise RuntimeError(f"Could not patch train/test split in {path}")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo.expanduser().resolve()
    oxe = root / "prismatic/vla/datasets/rlds/oxe"
    paths = {
        "config": oxe / "configs.py",
        "transform": oxe / "transforms.py",
        "mixture": oxe / "mixtures.py",
        "constants": root / "prismatic/vla/constants.py",
        "dataset": root / "prismatic/vla/datasets/rlds/dataset.py",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        backup_once(path)

    patch_config(paths["config"])
    patch_transform(paths["transform"])
    patch_mixture(paths["mixture"])
    patch_constants(paths["constants"])
    patch_official_splits(paths["dataset"])

    for path in paths.values():
        py_compile.compile(str(path), doraise=True)
        print("Patched and compiled:", path)
    print("OPENVLA_OFT_D3_REGISTRATION_INSTALLED")


if __name__ == "__main__":
    main()
