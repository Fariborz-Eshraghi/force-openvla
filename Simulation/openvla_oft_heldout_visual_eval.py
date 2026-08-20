#!/usr/bin/env python3
"""Teacher-forced visual evaluation for Panda D3 OpenVLA-OFT checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


EXPERT_COLOR = (24, 160, 88)
MODEL_COLOR = (220, 58, 62)
TEXT_COLOR = (235, 238, 242)
HUD_COLOR = (22, 25, 29)
AXIS_COLORS = ("#376fd0", "#a0459b", "#e08b2c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--openvla-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-indices", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum teacher-forced query positions per selected episode.",
    )
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def decode_instruction(value: object) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip().lower()


def measured_proprio(step: dict[str, object]) -> np.ndarray:
    observation = step["observation"]
    joints = np.asarray(observation["joint_qpos"], dtype=np.float32).reshape(-1)
    gripper = np.asarray(
        observation["gripper_open_fraction"], dtype=np.float32
    ).reshape(-1)
    proprio = np.concatenate([joints, gripper]).astype(np.float32)
    if proprio.shape != (8,) or not np.all(np.isfinite(proprio)):
        raise ValueError(f"Invalid measured proprio: shape={proprio.shape}, value={proprio}")
    if not 0.0 <= float(proprio[-1]) <= 1.0:
        raise ValueError(f"Invalid measured gripper fraction: {proprio[-1]}")
    return proprio


def phase_boundaries(actions: np.ndarray, cube_xyz: np.ndarray) -> dict[str, int]:
    count = len(actions)
    gripper = actions[:, 6]
    closed = np.flatnonzero(gripper <= 0.5)
    close_start = int(closed[0]) if len(closed) else max(1, count // 3)
    close_start = int(np.clip(close_start, 0, max(0, count - 1)))

    reopened = np.flatnonzero((np.arange(count) > close_start) & (gripper > 0.5))
    release_start = int(reopened[0]) if len(reopened) else count - 1

    base_z = float(np.median(cube_xyz[: max(1, min(5, count)), 2]))
    lifted = np.flatnonzero(
        (np.arange(count) >= close_start)
        & (np.arange(count) < release_start)
        & (cube_xyz[:, 2] >= base_z + 0.012)
    )
    lift_start = int(lifted[0]) if len(lifted) else min(close_start + 1, release_start)
    return {
        "close_start": close_start,
        "lift_start": lift_start,
        "release_start": release_start,
    }


def phase_name(step_index: int, bounds: dict[str, int]) -> str:
    if step_index < bounds["close_start"]:
        return "APPROACH"
    if step_index < bounds["lift_start"]:
        return "GRASP"
    if step_index < bounds["release_start"]:
        return "LIFT / TRANSPORT"
    return "RELEASE"


def add_phase_spans(ax: plt.Axes, bounds: dict[str, int], count: int) -> None:
    spans = (
        (0, bounds["close_start"], "#d9ead3"),
        (bounds["close_start"], bounds["lift_start"], "#fff2cc"),
        (bounds["lift_start"], bounds["release_start"], "#d9eaf7"),
        (bounds["release_start"], count - 1, "#eadcf1"),
    )
    for start, end, color in spans:
        if end > start:
            ax.axvspan(start, end, color=color, alpha=0.35, linewidth=0)


def gripper_state_mismatch(expert: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    class_differs = (expert > 0.5) != (predicted > 0.5)
    return class_differs & (np.abs(expert - predicted) >= 0.10)


def save_command_plot(
    output_path: Path,
    episode_index: int,
    expert: np.ndarray,
    predicted: np.ndarray,
    bounds: dict[str, int],
) -> None:
    steps = np.arange(len(expert))
    figure, axes = plt.subplots(
        3, 1, figsize=(14, 12), sharex=True, constrained_layout=True
    )
    figure.suptitle(
        f"Held-out episode {episode_index:03d}: current expert action vs OFT chunk horizon 0",
        fontsize=15,
    )

    axis = axes[0]
    for index, (name, color) in enumerate(zip(("X", "Y", "Z"), AXIS_COLORS)):
        mae = float(np.mean(np.abs(expert[:, index] - predicted[:, index])) * 1000.0)
        axis.plot(
            steps,
            expert[:, index] * 1000.0,
            color=color,
            linewidth=2.2,
            label=f"Test demo delta {name}",
        )
        axis.plot(
            steps,
            predicted[:, index] * 1000.0,
            color=color,
            linewidth=1.7,
            linestyle="--",
            label=f"OFT h0 delta {name} (MAE {mae:.2f} mm)",
        )
    add_phase_spans(axis, bounds, len(steps))
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_title("Translation commands")
    axis.set_ylabel("Millimetres / step")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)

    axis = axes[1]
    expert_rotation = np.rad2deg(expert[:, 3:6])
    predicted_rotation = np.rad2deg(predicted[:, 3:6])
    for index, (name, color) in enumerate(
        zip(("Roll", "Pitch", "Yaw"), AXIS_COLORS)
    ):
        mae = float(np.mean(np.abs(expert_rotation[:, index] - predicted_rotation[:, index])))
        axis.plot(
            steps,
            expert_rotation[:, index],
            color=color,
            linewidth=2.2,
            label=f"Test demo delta {name}",
        )
        axis.plot(
            steps,
            predicted_rotation[:, index],
            color=color,
            linewidth=1.7,
            linestyle="--",
            label=f"OFT h0 delta {name} (MAE {mae:.3f} deg)",
        )
    add_phase_spans(axis, bounds, len(steps))
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_title("World-frame rotation-vector commands")
    axis.set_ylabel("Degrees / step")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)

    axis = axes[2]
    axis.plot(steps, expert[:, 6], color="#18a058", linewidth=2.3, label="Test demo gripper")
    axis.plot(
        steps,
        predicted[:, 6],
        color="#dc3a3e",
        linewidth=1.8,
        linestyle="--",
        label="OFT h0 gripper",
    )
    axis.axhline(0.5, color="#20242a", linewidth=1, linestyle=":", label="Half-open reference")
    mismatch = gripper_state_mismatch(expert[:, 6], predicted[:, 6])
    axis.fill_between(
        steps,
        0,
        1,
        where=mismatch,
        color="#dc3a3e",
        alpha=0.12,
        step="mid",
        label="Class mismatch",
    )
    add_phase_spans(axis, bounds, len(steps))
    axis.set_title("Continuous gripper command")
    axis.set_xlabel("Teacher-forced demonstration step (10 Hz)")
    axis.set_ylabel("Open fraction")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_chunk_plot(
    output_path: Path,
    episode_index: int,
    expert_chunks: np.ndarray,
    predicted_chunks: np.ndarray,
) -> dict[str, object]:
    absolute_error = np.abs(expert_chunks - predicted_chunks)
    mae = np.mean(absolute_error, axis=0)
    horizons = np.arange(mae.shape[0])

    figure, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    figure.suptitle(
        f"Held-out episode {episode_index:03d}: OFT 8-step forecast error by horizon",
        fontsize=15,
    )

    for index, (name, color) in enumerate(zip(("X", "Y", "Z"), AXIS_COLORS)):
        axes[0].plot(
            horizons,
            mae[:, index] * 1000.0,
            marker="o",
            color=color,
            linewidth=2.0,
            label=f"delta {name}",
        )
    axes[0].set(title="Translation MAE", ylabel="Millimetres")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=3)

    for index, (name, color) in enumerate(zip(("Roll", "Pitch", "Yaw"), AXIS_COLORS)):
        axes[1].plot(
            horizons,
            np.rad2deg(mae[:, index + 3]),
            marker="o",
            color=color,
            linewidth=2.0,
            label=name,
        )
    axes[1].set(title="Rotation-vector component MAE", ylabel="Degrees")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3)

    axes[2].plot(
        horizons,
        mae[:, 6],
        marker="o",
        color="#18a058",
        linewidth=2.2,
        label="Gripper open fraction",
    )
    axes[2].set(
        title="Continuous gripper MAE",
        xlabel="Forecast horizon inside predicted chunk (0=current action)",
        ylabel="Open fraction",
        xticks=horizons,
    )
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    for axis in axes[:2]:
        axis.set_xticks(horizons)

    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return {
        "translation_mae_m_by_horizon": mae[:, :3].tolist(),
        "rotation_mae_rad_by_horizon": mae[:, 3:6].tolist(),
        "gripper_mae_by_horizon": mae[:, 6].tolist(),
    }


def save_proprio_plot(
    output_path: Path,
    episode_index: int,
    proprio: np.ndarray,
) -> None:
    steps = np.arange(len(proprio))
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    figure.suptitle(
        f"Held-out episode {episode_index:03d}: measured 8D proprio supplied to OFT",
        fontsize=15,
    )

    for joint_index in range(7):
        axes[0].plot(
            steps,
            proprio[:, joint_index],
            linewidth=1.8,
            label=f"q{joint_index + 1}",
        )
    axes[0].set(title="Measured Panda joint positions", ylabel="Radians")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)

    axes[1].plot(
        steps,
        proprio[:, 7],
        color="#18a058",
        linewidth=2.2,
        label="Measured gripper open fraction",
    )
    axes[1].set(
        title="Measured gripper state",
        xlabel="Teacher-forced demonstration step (10 Hz)",
        ylabel="Open fraction",
        ylim=(-0.05, 1.05),
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def contact_indices(count: int, bounds: dict[str, int]) -> list[int]:
    indices = set(np.linspace(0, count - 1, min(8, count), dtype=int).tolist())
    for value in (
        bounds["close_start"] - 1,
        bounds["close_start"],
        bounds["lift_start"],
        bounds["release_start"] - 1,
        bounds["release_start"],
    ):
        indices.add(int(np.clip(value, 0, count - 1)))
    return sorted(indices)


def save_contact_sheet(
    output_path: Path,
    images: list[Image.Image],
    expert: np.ndarray,
    predicted: np.ndarray,
    proprio: np.ndarray,
    bounds: dict[str, int],
) -> None:
    chosen = contact_indices(len(images), bounds)
    columns = 3
    rows = int(np.ceil(len(chosen) / columns))
    tile_w, image_h, text_h = 320, 240, 108
    sheet = Image.new(
        "RGB", (columns * tile_w, rows * (image_h + text_h)), (245, 246, 248)
    )
    font = load_font(14)
    small_font = load_font(12)
    resampling = getattr(Image, "Resampling", Image).LANCZOS

    for tile_index, step_index in enumerate(chosen):
        row, column = divmod(tile_index, columns)
        x0, y0 = column * tile_w, row * (image_h + text_h)
        frame = images[step_index].resize((tile_w, image_h), resampling)
        sheet.paste(frame, (x0, y0))
        draw = ImageDraw.Draw(sheet)
        exp = expert[step_index]
        pred = predicted[step_index]
        mismatch = bool(gripper_state_mismatch(exp[6], pred[6]))
        border = MODEL_COLOR if mismatch else EXPERT_COLOR
        draw.rectangle(
            (x0 + 1, y0 + 1, x0 + tile_w - 2, y0 + image_h - 2),
            outline=border,
            width=4,
        )
        draw.text(
            (x0 + 8, y0 + image_h + 5),
            f"step {step_index:02d}  {phase_name(step_index, bounds)}",
            fill=(25, 28, 32),
            font=font,
        )
        draw.text(
            (x0 + 8, y0 + image_h + 29),
            f"expert dxyz: {exp[0]:+.3f} {exp[1]:+.3f} {exp[2]:+.3f}",
            fill=EXPERT_COLOR,
            font=small_font,
        )
        draw.text(
            (x0 + 8, y0 + image_h + 48),
            f"OFT h0 dxyz: {pred[0]:+.3f} {pred[1]:+.3f} {pred[2]:+.3f}",
            fill=MODEL_COLOR,
            font=small_font,
        )
        draw.text(
            (x0 + 8, y0 + image_h + 67),
            f"gripper expert/model: {exp[6]:.2f} / {pred[6]:.2f}",
            fill=(25, 28, 32),
            font=small_font,
        )
        draw.text(
            (x0 + 8, y0 + image_h + 86),
            f"input proprio gripper: {proprio[step_index, 7]:.2f}",
            fill=(25, 28, 32),
            font=small_font,
        )

    sheet.save(output_path, quality=95)


def draw_command_bar(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    y: int,
    value: float,
    scale: float,
    color: tuple[int, int, int],
) -> None:
    half_width = 72
    draw.line(
        (center_x - half_width, y, center_x + half_width, y),
        fill=(92, 98, 105),
        width=1,
    )
    endpoint = center_x + int(np.clip(value / scale, -1, 1) * half_width)
    draw.line((center_x, y, endpoint, y), fill=color, width=6)


def save_video(
    output_path: Path,
    episode_index: int,
    images: list[Image.Image],
    expert: np.ndarray,
    predicted: np.ndarray,
    expert_chunks: np.ndarray,
    predicted_chunks: np.ndarray,
    proprio: np.ndarray,
    bounds: dict[str, int],
    fps: int,
) -> None:
    font = load_font(15)
    small_font = load_font(12)
    width, image_h, header_h, hud_h = 640, 480, 42, 196
    count = len(images)
    translation_scale = float(
        max(
            0.005,
            np.percentile(
                np.abs(np.concatenate([expert[:, :3], predicted[:, :3]])), 97
            ),
        )
    )
    gripper_mismatch = gripper_state_mismatch(expert[:, 6], predicted[:, 6])
    resampling = getattr(Image, "Resampling", Image).LANCZOS

    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=2,
    ) as writer:
        for step_index, source in enumerate(images):
            canvas = Image.new("RGB", (width, header_h + image_h + hud_h), HUD_COLOR)
            canvas.paste(source.resize((width, image_h), resampling), (0, header_h))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (14, 11),
                (
                    f"held-out episode {episode_index:03d}   step {step_index:02d}/{count - 1:02d}   "
                    f"{phase_name(step_index, bounds)}"
                ),
                fill=TEXT_COLOR,
                font=font,
            )

            exp = expert[step_index]
            pred = predicted[step_index]
            hud_y = header_h + image_h
            draw.text(
                (14, hud_y + 9),
                "horizon-0 translation (green: expert, red: OFT)",
                fill=TEXT_COLOR,
                font=small_font,
            )
            centers = (112, 320, 528)
            for axis_index, (axis_name, center_x) in enumerate(
                zip(("dx", "dy", "dz"), centers)
            ):
                draw.text(
                    (center_x - 10, hud_y + 32), axis_name, fill=TEXT_COLOR, font=small_font
                )
                draw_command_bar(
                    draw,
                    center_x,
                    hud_y + 56,
                    exp[axis_index],
                    translation_scale,
                    EXPERT_COLOR,
                )
                draw_command_bar(
                    draw,
                    center_x,
                    hud_y + 67,
                    pred[axis_index],
                    translation_scale,
                    MODEL_COLOR,
                )

            chunk_error = np.abs(expert_chunks[step_index] - predicted_chunks[step_index])
            chunk_translation_mae_mm = float(np.mean(chunk_error[:, :3]) * 1000.0)
            chunk_rotation_mae_deg = float(np.rad2deg(np.mean(chunk_error[:, 3:6])))
            chunk_gripper_mae = float(np.mean(chunk_error[:, 6]))
            draw.text(
                (14, hud_y + 82),
                f"gripper expert/OFT h0: {exp[6]:.2f} / {pred[6]:.2f}",
                fill=TEXT_COLOR,
                font=small_font,
            )
            draw.text(
                (328, hud_y + 82),
                f"measured proprio gripper: {proprio[step_index, 7]:.2f}",
                fill=TEXT_COLOR,
                font=small_font,
            )
            draw.text(
                (14, hud_y + 104),
                (
                    "full 8-step chunk MAE: "
                    f"translation {chunk_translation_mae_mm:.2f} mm | "
                    f"rotation {chunk_rotation_mae_deg:.3f} deg | "
                    f"gripper {chunk_gripper_mae:.3f}"
                ),
                fill=(180, 185, 190),
                font=small_font,
            )

            timeline_x0, timeline_x1, timeline_y = 14, width - 14, hud_y + 145
            for timeline_step in range(count):
                x_start = timeline_x0 + int(
                    (timeline_x1 - timeline_x0) * timeline_step / count
                )
                x_end = timeline_x0 + int(
                    (timeline_x1 - timeline_x0) * (timeline_step + 1) / count
                )
                color = MODEL_COLOR if gripper_mismatch[timeline_step] else (95, 105, 115)
                draw.rectangle(
                    (x_start, timeline_y, max(x_start + 1, x_end), timeline_y + 9),
                    fill=color,
                )
            marker_x = timeline_x0 + int(
                (timeline_x1 - timeline_x0) * step_index / max(1, count - 1)
            )
            draw.line(
                (marker_x, timeline_y - 5, marker_x, timeline_y + 14),
                fill=(255, 255, 255),
                width=2,
            )
            draw.text(
                (14, hud_y + 164),
                "red timeline = expert/OFT horizon-0 gripper-state mismatch",
                fill=(180, 185, 190),
                font=small_font,
            )
            writer.append_data(np.asarray(canvas))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.openvla_repo = args.openvla_repo.expanduser().resolve()
    if not args.openvla_repo.is_dir():
        raise FileNotFoundError(f"OpenVLA-OFT repository not found: {args.openvla_repo}")

    required_checkpoint_files = (
        "config.json",
        "dataset_statistics.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    )
    missing = [
        name for name in required_checkpoint_files if not (args.checkpoint / name).is_file()
    ]
    if not any(args.checkpoint.glob("action_head--*checkpoint.pt")):
        missing.append("action_head--*checkpoint.pt")
    if not any(args.checkpoint.glob("proprio_projector--*checkpoint.pt")):
        missing.append("proprio_projector--*checkpoint.pt")
    if missing:
        raise FileNotFoundError(
            f"Incomplete OFT checkpoint {args.checkpoint}; missing: {', '.join(missing)}"
        )

    sys.argv.append("panda_d3")
    sys.path.insert(0, str(args.openvla_repo))

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    import torch

    from experiments.robot.openvla_utils import (
        get_action_head,
        get_processor,
        get_proprio_projector,
        get_vla,
        get_vla_action,
    )
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM

    if (NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM) != (8, 7, 8):
        raise RuntimeError(
            "Expected Panda D3 constants (chunk=8, action=7, proprio=8), got "
            f"{(NUM_ACTIONS_CHUNK, ACTION_DIM, PROPRIO_DIM)}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the OpenVLA-OFT evaluator subprocess.")

    requested = sorted(set(args.episode_indices))
    if not requested or requested[0] < 0:
        raise ValueError("Episode indices must be non-negative.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    print(f"Loading test split: {args.data_root / args.dataset_name}", flush=True)
    dataset = tfds.load(
        args.dataset_name,
        split="test",
        data_dir=str(args.data_root),
        shuffle_files=False,
    )

    cfg = SimpleNamespace(
        pretrained_checkpoint=str(args.checkpoint),
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=1,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        lora_rank=32,
        unnorm_key=args.dataset_name,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50,
    )

    print(f"Loading merged OFT checkpoint: {args.checkpoint}", flush=True)
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, PROPRIO_DIM)

    with (args.checkpoint / "dataset_statistics.json").open(
        "r", encoding="utf-8"
    ) as handle:
        dataset_statistics = json.load(handle)
    if args.dataset_name not in dataset_statistics:
        raise KeyError(f"Missing {args.dataset_name!r} in dataset_statistics.json")
    proprio_stats = dataset_statistics[args.dataset_name].get("proprio", {})
    if np.asarray(proprio_stats.get("q01", [])).shape != (8,):
        raise ValueError("Checkpoint does not contain 8D proprio normalization statistics.")

    summaries: list[dict[str, object]] = []
    found: set[int] = set()
    for episode_index, episode in enumerate(tfds.as_numpy(dataset)):
        if episode_index not in requested:
            if episode_index > requested[-1]:
                break
            continue

        found.add(episode_index)
        all_steps = list(episode["steps"])
        valid_query_count = len(all_steps) - NUM_ACTIONS_CHUNK + 1
        if valid_query_count <= 0:
            raise RuntimeError(
                f"Held-out episode {episode_index} has {len(all_steps)} actions, fewer than "
                f"the {NUM_ACTIONS_CHUNK}-step OFT chunk."
            )
        query_count = valid_query_count
        if args.max_steps is not None:
            query_count = min(query_count, args.max_steps)
        query_steps = all_steps[:query_count]

        images = [
            Image.fromarray(step["observation"]["image"]).convert("RGB")
            for step in query_steps
        ]
        all_expert_actions = np.stack(
            [np.asarray(step["action"], dtype=np.float32) for step in all_steps]
        )
        expert_chunks = np.stack(
            [
                all_expert_actions[index : index + NUM_ACTIONS_CHUNK]
                for index in range(query_count)
            ]
        )
        proprio = np.stack([measured_proprio(step) for step in query_steps])
        cube_xyz = np.stack(
            [
                np.asarray(step["observation"]["cube_pose"][:3], dtype=np.float32)
                for step in query_steps
            ]
        )
        instruction = decode_instruction(query_steps[0]["language_instruction"])

        predicted_chunks: list[np.ndarray] = []
        started = time.time()
        for step_index, (image, state) in enumerate(zip(images, proprio)):
            observation = {
                "full_image": np.asarray(image),
                "state": state.copy(),
            }
            actions = np.asarray(
                get_vla_action(
                    cfg,
                    vla,
                    processor,
                    observation,
                    instruction,
                    action_head,
                    proprio_projector,
                ),
                dtype=np.float32,
            )
            if actions.shape != (NUM_ACTIONS_CHUNK, ACTION_DIM):
                raise ValueError(
                    f"Step {step_index}: expected {(NUM_ACTIONS_CHUNK, ACTION_DIM)}, "
                    f"got {actions.shape}."
                )
            if not np.all(np.isfinite(actions)):
                raise ValueError(f"Step {step_index}: prediction contains NaN or infinity.")
            predicted_chunks.append(actions)
            if step_index == 0 or (step_index + 1) % 10 == 0 or step_index + 1 == query_count:
                print(
                    f"episode {episode_index:03d}: predicted {step_index + 1}/{query_count} chunks",
                    flush=True,
                )

        predicted_chunks_array = np.stack(predicted_chunks)
        expert_h0 = expert_chunks[:, 0, :]
        predicted_h0 = predicted_chunks_array[:, 0, :]
        bounds = phase_boundaries(expert_h0, cube_xyz)
        prefix = f"episode_{episode_index:03d}"
        command_plot = args.output_dir / f"{prefix}_commands.png"
        chunk_plot = args.output_dir / f"{prefix}_chunk_horizons.png"
        proprio_plot = args.output_dir / f"{prefix}_proprio.png"
        contact_sheet = args.output_dir / f"{prefix}_contact_sheet.png"
        video = args.output_dir / f"{prefix}_overlay.mp4"
        predictions_file = args.output_dir / f"{prefix}_predictions.npz"

        save_command_plot(command_plot, episode_index, expert_h0, predicted_h0, bounds)
        chunk_metrics = save_chunk_plot(
            chunk_plot,
            episode_index,
            expert_chunks,
            predicted_chunks_array,
        )
        save_proprio_plot(proprio_plot, episode_index, proprio)
        save_contact_sheet(
            contact_sheet,
            images,
            expert_h0,
            predicted_h0,
            proprio,
            bounds,
        )
        save_video(
            video,
            episode_index,
            images,
            expert_h0,
            predicted_h0,
            expert_chunks,
            predicted_chunks_array,
            proprio,
            bounds,
            args.fps,
        )
        np.savez_compressed(
            predictions_file,
            expert_actions=expert_h0,
            predicted_actions=predicted_h0,
            expert_action_chunks=expert_chunks,
            predicted_action_chunks=predicted_chunks_array,
            proprio=proprio,
            cube_xyz=cube_xyz,
        )

        h0_mae = np.mean(np.abs(expert_h0 - predicted_h0), axis=0)
        summary = {
            "episode_index": episode_index,
            "raw_episode_steps": len(all_steps),
            "valid_chunk_queries": valid_query_count,
            "evaluated_queries": query_count,
            "chunk_shape": [NUM_ACTIONS_CHUNK, ACTION_DIM],
            "proprio_shape": [PROPRIO_DIM],
            "instruction": instruction,
            "phase_boundaries": bounds,
            "elapsed_seconds": round(time.time() - started, 2),
            "horizon_0_mae": {
                "translation_m": h0_mae[:3].tolist(),
                "rotation_rad": h0_mae[3:6].tolist(),
                "gripper_open_fraction": float(h0_mae[6]),
            },
            "chunk_metrics": chunk_metrics,
            "commands_plot": str(command_plot.resolve()),
            "chunk_plot": str(chunk_plot.resolve()),
            "proprio_plot": str(proprio_plot.resolve()),
            "contact_sheet": str(contact_sheet.resolve()),
            "video": str(video.resolve()),
            "predictions": str(predictions_file.resolve()),
        }
        summaries.append(summary)
        print(f"Saved OFT visual diagnostics for episode {episode_index:03d}", flush=True)

    missing_indices = sorted(set(requested) - found)
    if missing_indices:
        raise IndexError(f"Test episode indices do not exist: {missing_indices}")

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "dataset": args.dataset_name,
                "split": "test",
                "camera": "observation.image (third-person)",
                "teacher_forced": True,
                "input": "third-person RGB + language + measured 8D joint/gripper proprio",
                "output": "8 future actions x 7 continuous dimensions",
                "episodes": summaries,
            },
            handle,
            indent=2,
        )
    print(f"SUMMARY_JSON={summary_path.resolve()}", flush=True)
    print("OPENVLA_OFT_HELDOUT_VISUAL_EVAL_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
