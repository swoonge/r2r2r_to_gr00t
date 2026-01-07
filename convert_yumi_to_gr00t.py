#!/usr/bin/env python
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert r2r2r YuMi data to GR00T LeRobot v2 format.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/vision/Sim2Real_Data_Augmentation_for_VLA/yumi_coffee_maker/successes"),
        help="Root folder containing env_* episode directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/vision/Sim2Real_Data_Augmentation_for_VLA/r2r2r_to_gr00t/converted_yumi"),
        help="Output dataset root.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="put the white cup on the coffee machine",
        help="Language instruction for all episodes.",
    )
    parser.add_argument("--fps", type=int, default=15, help="Video fps and timestamp rate.")
    parser.add_argument(
        "--resize",
        type=str,
        default="640x480",
        help=(
            "Resize to WIDTHxHEIGHT with padding to preserve aspect ratio. "
            "Use 0/keep to keep original size. Single int means square."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for LeRobot.")
    parser.add_argument(
        "--camera-keys",
        nargs=2,
        default=["camera_0", "camera_1"],
        help="Camera folder names under each episode directory.",
    )
    parser.add_argument(
        "--camera-names",
        nargs=2,
        default=["front", "wrist"],
        help="Output camera names used in GR00T meta/modality.json.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit number of episodes for quick tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-root if it exists.",
    )
    return parser.parse_args()


def parse_resize_arg(resize: str) -> tuple[int | None, int | None]:
    if resize is None:
        return None, None
    value = resize.strip().lower()
    if value in {"0", "keep", "none", "original"}:
        return None, None
    if "x" in value:
        parts = value.split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid resize format: {resize}")
        width = int(parts[0])
        height = int(parts[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"Resize dimensions must be positive: {resize}")
        return width, height
    size = int(value)
    if size <= 0:
        return None, None
    return size, size


def list_episode_dirs(input_root: Path) -> list[Path]:
    episode_dirs = []
    for path in sorted(input_root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "robot_data" / "robot_data.h5").exists():
            episode_dirs.append(path)
    return episode_dirs


def build_joint_order(joint_names: list[str]) -> tuple[list[int], list[str]]:
    left_idxs = [
        i for i, name in enumerate(joint_names) if name.endswith("_l") and "gripper" not in name
    ]
    right_idxs = [
        i for i, name in enumerate(joint_names) if name.endswith("_r") and "gripper" not in name
    ]
    gripper_idxs = [i for i, name in enumerate(joint_names) if "gripper" in name]
    ordered = left_idxs + right_idxs + gripper_idxs
    ordered_names = [joint_names[i] for i in ordered]
    return ordered, ordered_names


def load_joint_names(joint_names_path: Path) -> list[str]:
    with open(joint_names_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_image_paths(camera_dir: Path) -> list[Path]:
    images = sorted(camera_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    return images


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ffmpeg_encode(
    image_dir: Path,
    output_path: Path,
    fps: int,
    frame_count: int,
    resize_width: int | None,
    resize_height: int | None,
) -> None:
    input_pattern = str(image_dir / "%04d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        input_pattern,
        "-frames:v",
        str(frame_count),
    ]
    if resize_width is not None and resize_height is not None:
        filter_chain = (
            f"scale={resize_width}:{resize_height}:force_original_aspect_ratio=decrease,"
            f"pad={resize_width}:{resize_height}:(ow-iw)/2:(oh-ih)/2"
        )
        cmd += ["-vf", filter_chain]
    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if len(args.camera_keys) != len(args.camera_names):
        raise ValueError("camera-keys and camera-names must have same length")

    episode_dirs = list_episode_dirs(args.input_root)
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]

    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found in {args.input_root}")

    ensure_empty_dir(args.output_root, args.overwrite)
    data_dir = args.output_root / "data"
    meta_dir = args.output_root / "meta"
    videos_dir = args.output_root / "videos"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Joint order and names from first episode
    first_joint_names = load_joint_names(episode_dirs[0] / "robot_data" / "joint_names.txt")
    joint_order, ordered_joint_names = build_joint_order(first_joint_names)

    # Determine image shape
    sample_image = (
        episode_dirs[0] / args.camera_keys[0] / "rgb" / "0000.jpg"
    )
    with Image.open(sample_image) as img:
        orig_width, orig_height = img.size

    resize_width, resize_height = parse_resize_arg(args.resize)
    if resize_width is None or resize_height is None:
        image_width = orig_width
        image_height = orig_height
    else:
        image_width = resize_width
        image_height = resize_height

    # Prepare chunk folders
    num_chunks = (len(episode_dirs) + args.chunk_size - 1) // args.chunk_size
    for i in range(num_chunks):
        (data_dir / f"chunk-{i:03d}").mkdir(parents=True, exist_ok=True)

    global_index = 0
    episodes_meta = []

    for episode_index, episode_dir in enumerate(tqdm(episode_dirs, desc="Episodes")):
        h5_path = episode_dir / "robot_data" / "robot_data.h5"
        with h5py.File(h5_path, "r") as f:
            joint_angles = f["joint_angles"][:].astype(np.float32)

        joint_angles = joint_angles[:, joint_order]
        total_frames = joint_angles.shape[0]
        if total_frames < 2:
            continue

        camera_image_paths = {}
        for cam_key in args.camera_keys:
            cam_dir = episode_dir / cam_key / "rgb"
            if not cam_dir.exists():
                raise FileNotFoundError(f"Missing camera directory: {cam_dir}")
            camera_image_paths[cam_key] = load_image_paths(cam_dir)

        for cam_key, images in camera_image_paths.items():
            if len(images) != total_frames:
                raise ValueError(
                    f"Frame count mismatch in {episode_dir} for {cam_key}: "
                    f"{len(images)} vs {total_frames}"
                )

        seq_length = total_frames - 1
        chunk_id = episode_index // args.chunk_size
        parquet_path = data_dir / f"chunk-{chunk_id:03d}" / f"episode_{episode_index:06d}.parquet"

        records = []
        for step in range(seq_length):
            state_t = joint_angles[step]
            action_t = joint_angles[step + 1]
            records.append(
                {
                    "observation.state": state_t,
                    "action": action_t,
                    "timestamp": np.float32(step / args.fps),
                    "frame_index": np.int64(step),
                    "episode_index": np.int64(episode_index),
                    "index": np.int64(global_index),
                    "task_index": np.int64(0),
                }
            )
            global_index += 1

        pd.DataFrame(records).to_parquet(parquet_path, index=False)

        # Encode videos
        for cam_key, cam_name in zip(args.camera_keys, args.camera_names):
            video_dir = videos_dir / f"chunk-{chunk_id:03d}" / f"observation.images.{cam_name}"
            video_dir.mkdir(parents=True, exist_ok=True)
            output_path = video_dir / f"episode_{episode_index:06d}.mp4"
            ffmpeg_encode(
                image_dir=episode_dir / cam_key / "rgb",
                output_path=output_path,
                fps=args.fps,
                frame_count=seq_length,
                resize_width=resize_width,
                resize_height=resize_height,
            )

        episodes_meta.append(
            {"episode_index": episode_index, "tasks": [args.task], "length": seq_length}
        )

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": args.task}) + "\n")

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    # info.json
    features = {
        "observation.state": {
            "dtype": "float32",
            "names": ordered_joint_names,
            "shape": [len(ordered_joint_names)],
        },
        "action": {
            "dtype": "float32",
            "names": ordered_joint_names,
            "shape": [len(ordered_joint_names)],
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }

    for cam_name in args.camera_names:
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [image_height, image_width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": image_height,
                "video.width": image_width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": args.fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "yumi",
        "total_episodes": len(episodes_meta),
        "total_frames": sum(ep["length"] for ep in episodes_meta),
        "total_tasks": 1,
        "total_chunks": num_chunks,
        "chunks_size": args.chunk_size,
        "fps": args.fps,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "total_videos": len(args.camera_names) * len(episodes_meta),
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # modality.json
    modality = {
        "state": {
            "left_arm": {"start": 0, "end": 7},
            "right_arm": {"start": 7, "end": 14},
            "gripper": {"start": 14, "end": 16},
        },
        "action": {
            "left_arm": {"start": 0, "end": 7},
            "right_arm": {"start": 7, "end": 14},
            "gripper": {"start": 14, "end": 16},
        },
        "video": {
            cam_name: {"original_key": f"observation.images.{cam_name}"}
            for cam_name in args.camera_names
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)


if __name__ == "__main__":
    main()
