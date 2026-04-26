"""
Convert LIBERO create_dataset HDF5 files → LeRobot v2.1 dataset format.

Source layout (one suite, e.g. libero_spatial):
    <input_dir>/
        pick_up_the_black_bowl_..._demo.hdf5
        pick_up_the_black_bowl_next_..._demo.hdf5
        ...

Output layout:
    <output_dir>/
        meta/
            info.json
            tasks.jsonl
            episodes.jsonl
            episodes_stats.jsonl
        data/
            chunk-000/
                episode_000000.parquet
                episode_000001.parquet
                ...
        videos/
            chunk-000/
                observation.images.agentview_rgb/
                    episode_000000.mp4  ...
                observation.images.eye_in_hand_rgb/
                    episode_000000.mp4  ...
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

FPS = 20
CHUNKS_SIZE = 1000          # max episodes per chunk (standard lerobot value)

# ── AV1 codec auto-detection ──────────────────────────────────────────────────
def _detect_video_codec() -> tuple:
    """
    Return (ffmpeg_codec_name, info_json_name, extra_args) for the best
    AV1-capable encoder available, falling back to libx265 then libx264.
    """
    candidates = [
        ("libsvtav1",  "av1",  ["-crf", "35", "-preset", "6"]),
        ("libaom-av1", "av1",  ["-crf", "35", "-cpu-used", "6", "-row-mt", "1"]),
        ("libx265",    "hevc", ["-crf", "28", "-preset", "fast"]),
        ("libx264",    "h264", ["-crf", "23", "-preset", "fast"]),
    ]
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True
    )
    encoders_text = result.stdout + result.stderr
    for codec, label, extra in candidates:
        if codec in encoders_text:
            print(f"[info] Video codec selected: {codec} ({label})")
            return codec, label, extra
    raise RuntimeError("No suitable ffmpeg video encoder found (need libsvtav1/libaom-av1/libx265/libx264)")

_FFMPEG_CODEC, VIDEO_CODEC_INFO, _FFMPEG_EXTRA_ARGS = _detect_video_codec()

# camera key in HDF5  →  lerobot video key
CAMERA_MAP = {
    "agentview_rgb":    "observation.images.agentview_rgb",
    "eye_in_hand_rgb":  "observation.images.eye_in_hand_rgb",
}


def episode_chunk(episode_index: int) -> int:
    return episode_index // CHUNKS_SIZE


def chunk_str(chunk: int) -> str:
    return f"chunk-{chunk:03d}"


# ──────────────────────────────────────────────────────────────────────────────
# Video writing
# ──────────────────────────────────────────────────────────────────────────────

def write_video(frames: np.ndarray, path: Path, fps: int = FPS):
    """
    Write a (T, H, W, 3) uint8 RGB array to an .mp4 file using ffmpeg.
    Uses the best available codec: AV1 (libsvtav1/libaom-av1) → HEVC → H.264.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, _ = frames.shape
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", _FFMPEG_CODEC,
        *_FFMPEG_EXTRA_ARGS,
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in frames:
        proc.stdin.write(frame.tobytes())   # RGB bytes, ffmpeg expects rgb24
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed for {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Parquet writing
# ──────────────────────────────────────────────────────────────────────────────

HF_FEATURES = {
    "observation.state": {
        "feature": {"dtype": "float32", "_type": "Value"},
        "length": 8,
        "_type": "Sequence",
    },
    "action": {
        "feature": {"dtype": "float32", "_type": "Value"},
        "length": 7,
        "_type": "Sequence",
    },
    "timestamp":     {"dtype": "float32", "_type": "Value"},
    "frame_index":   {"dtype": "int64",   "_type": "Value"},
    "episode_index": {"dtype": "int64",   "_type": "Value"},
    "index":         {"dtype": "int64",   "_type": "Value"},
    "task_index":    {"dtype": "int64",   "_type": "Value"},
}

HF_META = json.dumps({"info": {"features": HF_FEATURES}})


def write_parquet(rows: dict, path: Path):
    """
    rows: dict of column_name → python list
    Writes a parquet file with the huggingface schema metadata expected by LeRobot.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    T = len(rows["timestamp"])

    # Build PyArrow arrays
    arrays = []
    fields = []

    # observation.state  (fixed_size_list<float32>[8])
    obs_state = pa.array(
        [row.tolist() for row in rows["observation.state"]],
        type=pa.list_(pa.float32(), 8),
    )
    arrays.append(obs_state)
    fields.append(pa.field("observation.state", pa.list_(pa.float32(), 8)))

    # action  (fixed_size_list<float32>[7])
    action = pa.array(
        [row.tolist() for row in rows["action"]],
        type=pa.list_(pa.float32(), 7),
    )
    arrays.append(action)
    fields.append(pa.field("action", pa.list_(pa.float32(), 7)))

    # scalar columns
    for col, arrow_type in [
        ("timestamp",     pa.float32()),
        ("frame_index",   pa.int64()),
        ("episode_index", pa.int64()),
        ("index",         pa.int64()),
        ("task_index",    pa.int64()),
    ]:
        arrays.append(pa.array(rows[col], type=arrow_type))
        fields.append(pa.field(col, arrow_type))

    schema = pa.schema(fields, metadata={b"huggingface": HF_META.encode()})
    table = pa.table({f.name: arr for f, arr in zip(fields, arrays)}, schema=schema)
    pq.write_table(table, str(path), compression="snappy")


# ──────────────────────────────────────────────────────────────────────────────
# Per-episode stats
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(rows: dict, episode_index: int) -> dict:
    """Compute min/max/mean/std/count for each numeric feature, matching lerobot format."""

    def scalar_stats(arr):
        arr = np.array(arr, dtype=np.float64)
        return {
            "min":   float(arr.min()),
            "max":   float(arr.max()),
            "mean":  float(arr.mean()),
            "std":   float(arr.std()),
            "count": [len(arr)],
        }

    def vector_stats(arr):
        arr = np.array(arr, dtype=np.float64)   # (T, D)
        return {
            "min":   arr.min(axis=0).tolist(),
            "max":   arr.max(axis=0).tolist(),
            "mean":  arr.mean(axis=0).tolist(),
            "std":   arr.std(axis=0).tolist(),
            "count": [len(arr)],
        }

    def image_stats(H, W, count):
        # images are normalised [0,1]; lerobot stores channel-wise (C,1,1) shaped stats
        return {
            "min":   [[[0.0]], [[0.0]], [[0.0]]],
            "max":   [[[1.0]], [[1.0]], [[1.0]]],
            "mean":  [[[0.5]],  [[0.5]],  [[0.5]]],
            "std":   [[[0.5]],  [[0.5]],  [[0.5]]],
            "count": [count],
        }

    T = len(rows["timestamp"])
    stats = {}

    # video features: placeholder channel-wise stats
    for cam_key in CAMERA_MAP.values():
        stats[cam_key] = image_stats(None, None, T)

    stats["observation.state"] = vector_stats(rows["observation.state"])
    stats["action"]            = vector_stats(rows["action"])
    stats["timestamp"]         = scalar_stats(rows["timestamp"])
    stats["frame_index"]       = scalar_stats(rows["frame_index"])
    stats["episode_index"]     = scalar_stats(rows["episode_index"])
    stats["index"]             = scalar_stats(rows["index"])
    stats["task_index"]        = scalar_stats(rows["task_index"])

    return {"episode_index": episode_index, "stats": stats}


# ──────────────────────────────────────────────────────────────────────────────
# info.json builder
# ──────────────────────────────────────────────────────────────────────────────

def build_info_json(
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
    total_chunks: int,
    img_height: int,
    img_width: int,
    robot_type: str = "franka",
) -> dict:

    video_feature = {
        "dtype": "video",
        "shape": [img_height, img_width, 3],
        "names": ["height", "width", "rgb"],
        "info": {
            "video.height":       img_height,
            "video.width":        img_width,
            "video.codec":        VIDEO_CODEC_INFO,
            "video.pix_fmt":      "yuv420p",
            "video.is_depth_map": False,
            "video.fps":          FPS,
            "video.channels":     3,
            "has_audio":          False,
        },
    }

    return {
        "codebase_version": "v2.1",
        "robot_type":       robot_type,
        "total_episodes":   total_episodes,
        "total_frames":     total_frames,
        "total_tasks":      total_tasks,
        "total_videos":     total_videos,
        "total_chunks":     total_chunks,
        "chunks_size":      CHUNKS_SIZE,
        "fps":              FPS,
        "splits":           {"train": f"0:{total_episodes}"},
        "data_path":  "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            **{lerobot_key: video_feature for lerobot_key in CAMERA_MAP.values()},
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": {
                    "motors": [
                        "x", "y", "z",
                        "axis_angle1", "axis_angle2", "axis_angle3",
                        "gripper", "gripper",
                    ]
                },
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": {
                    "motors": [
                        "x", "y", "z",
                        "axis_angle1", "axis_angle2", "axis_angle3",
                        "gripper",
                    ]
                },
            },
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
            "index":         {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main conversion logic
# ──────────────────────────────────────────────────────────────────────────────

def convert(input_dir: Path, output_dir: Path, robot_type: str = "franka"):
    """
    Convert all *_demo.hdf5 files under input_dir into a single LeRobot v2.1 dataset.
    """

    hdf5_files = sorted(input_dir.glob("*_demo.hdf5"))
    if not hdf5_files:
        print(f"[error] No *_demo.hdf5 files found in {input_dir}")
        sys.exit(1)
    print(f"[info] Found {len(hdf5_files)} HDF5 file(s) in {input_dir}")

    # Create output directories
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)

    # ── Pass 1: collect tasks (one task per HDF5 file) ──────────────────────
    tasks = []           # list of task strings, index = task_index
    task_to_idx = {}     # task_str → task_index

    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as f:
            problem_info = json.loads(f["data"].attrs["problem_info"])
            lang = problem_info["language_instruction"].strip()
        if lang not in task_to_idx:
            task_to_idx[lang] = len(tasks)
            tasks.append(lang)

    print(f"[info] Total tasks: {len(tasks)}")
    for i, t in enumerate(tasks):
        print(f"  [{i}] {t}")

    # ── Pass 2: iterate all episodes ────────────────────────────────────────
    global_episode_idx = 0
    global_frame_idx   = 0
    total_frames       = 0

    episodes_meta  = []   # for episodes.jsonl
    episodes_stats = []   # for episodes_stats.jsonl

    img_height = None
    img_width  = None

    for hdf5_path in hdf5_files:
        print(f"\n[info] Processing {hdf5_path.name}")
        with h5py.File(hdf5_path, "r") as f:
            problem_info  = json.loads(f["data"].attrs["problem_info"])
            lang          = problem_info["language_instruction"].strip()
            task_idx      = task_to_idx[lang]
            demo_keys     = sorted(f["data"].keys())

            for demo_key in demo_keys:
                ep = f[f"data/{demo_key}"]
                T  = ep["actions"].shape[0]

                # ── read arrays ────────────────────────────────────────────
                actions        = ep["actions"][()]               # (T, 7)
                ee_states      = ep["obs/ee_states"][()]         # (T, 6)
                gripper_states = ep["obs/gripper_states"][()]    # (T, 2)
                obs_state      = np.concatenate(
                    [ee_states, gripper_states], axis=1
                ).astype(np.float32)                             # (T, 8)

                agentview_rgb    = ep["obs/agentview_rgb"][()]   # (T, H, W, 3)
                eye_in_hand_rgb  = ep["obs/eye_in_hand_rgb"][()] # (T, H, W, 3)

                # Capture image size from first episode
                if img_height is None:
                    img_height = agentview_rgb.shape[1]
                    img_width  = agentview_rgb.shape[2]
                    print(f"[info] Image size detected: {img_height}x{img_width}")

                chunk      = episode_chunk(global_episode_idx)
                chunk_name = chunk_str(chunk)

                # ── write videos ───────────────────────────────────────────
                for hdf5_cam, lerobot_key in CAMERA_MAP.items():
                    frames = (
                        agentview_rgb   if hdf5_cam == "agentview_rgb"
                        else eye_in_hand_rgb
                    )
                    vid_path = (
                        output_dir / "videos" / chunk_name
                        / lerobot_key
                        / f"episode_{global_episode_idx:06d}.mp4"
                    )
                    write_video(frames, vid_path, fps=FPS)

                # ── build parquet rows ─────────────────────────────────────
                timestamps   = (np.arange(T) / FPS).astype(np.float32)
                frame_idxs   = np.arange(T, dtype=np.int64)
                episode_idxs = np.full(T, global_episode_idx, dtype=np.int64)
                global_idxs  = np.arange(
                    global_frame_idx, global_frame_idx + T, dtype=np.int64
                )
                task_idxs    = np.full(T, task_idx, dtype=np.int64)

                rows = {
                    "observation.state": obs_state,
                    "action":            actions.astype(np.float32),
                    "timestamp":         timestamps.tolist(),
                    "frame_index":       frame_idxs.tolist(),
                    "episode_index":     episode_idxs.tolist(),
                    "index":             global_idxs.tolist(),
                    "task_index":        task_idxs.tolist(),
                }

                parquet_path = (
                    output_dir / "data" / chunk_name
                    / f"episode_{global_episode_idx:06d}.parquet"
                )
                write_parquet(rows, parquet_path)

                # ── accumulate metadata ────────────────────────────────────
                episodes_meta.append({
                    "episode_index": global_episode_idx,
                    "tasks":         [lang],
                    "length":        T,
                })
                episodes_stats.append(
                    compute_stats(rows, global_episode_idx)
                )

                print(
                    f"  episode {global_episode_idx:06d}  "
                    f"(task_idx={task_idx}, T={T})  → {parquet_path.name}"
                )

                global_frame_idx   += T
                global_episode_idx += 1
                total_frames       += T

    total_episodes = global_episode_idx
    total_videos   = total_episodes * len(CAMERA_MAP)
    total_chunks   = episode_chunk(total_episodes - 1) + 1 if total_episodes > 0 else 1

    # ── write meta files ────────────────────────────────────────────────────
    meta_dir = output_dir / "meta"

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as fout:
        for i, task_str in enumerate(tasks):
            fout.write(json.dumps({"task_index": i, "task": task_str}) + "\n")

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as fout:
        for ep_meta in episodes_meta:
            fout.write(json.dumps(ep_meta) + "\n")

    # episodes_stats.jsonl
    with open(meta_dir / "episodes_stats.jsonl", "w") as fout:
        for ep_stat in episodes_stats:
            fout.write(json.dumps(ep_stat) + "\n")

    # info.json
    info = build_info_json(
        total_episodes = total_episodes,
        total_frames   = total_frames,
        total_tasks    = len(tasks),
        total_videos   = total_videos,
        total_chunks   = total_chunks,
        img_height     = img_height or 128,
        img_width      = img_width  or 128,
        robot_type     = robot_type,
    )
    with open(meta_dir / "info.json", "w") as fout:
        json.dump(info, fout, indent=4)

    print(f"\n[done] LeRobot dataset written to: {output_dir}")
    print(f"       total_episodes : {total_episodes}")
    print(f"       total_frames   : {total_frames}")
    print(f"       total_tasks    : {len(tasks)}")
    print(f"       total_videos   : {total_videos}")
    print(f"       total_chunks   : {total_chunks}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LIBERO HDF5 dataset(s) to LeRobot v2.1 format."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help=(
            "Directory containing one or more *_demo.hdf5 files for a single task suite "
            "(e.g. datasets/datasets/libero_spatial/)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root directory to write the LeRobot v2.1 dataset into.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="franka",
        help="Robot type string written into info.json (default: franka).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        input_dir  = Path(args.input_dir),
        output_dir = Path(args.output_dir),
        robot_type = args.robot_type,
    )
