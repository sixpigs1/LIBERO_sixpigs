#!/usr/bin/env python3
"""
Filter static (near-zero-action) frames from LIBERO demo.hdf5 files.

Supported formats
-----------------
* libero_std  – output of create_dataset.py
                Groups: data/demo_N/{actions, states, rewards, dones,
                        robot_states, obs/{agentview_rgb, eye_in_hand_rgb,
                        ee_states, gripper_states, joint_states, ...}}
                This is the CORRECT format to filter (images already rendered).

* intermediate – output of collect_demonstration.py
                Groups: data/demo_N/{actions, states}
                WARNING: filtering this format before create_dataset.py will
                cause playback divergence because removed frames break state
                continuity.  Prefer filtering the libero_std output.

The format is auto-detected from the HDF5 contents.

Strategy
--------
A frame at time-step t is "static" when
    ||action[t, :6]||₂  <  --action-threshold
(dims 0-5 are EEF position + rotation; dim 6 is the gripper command).

Consecutive static runs of length ≥ --min-static-run are collapsed: only the
FIRST frame of each run is kept.  The very first and last frames of every
episode are always kept.

Usage
-----
    python scripts/filter_static_frames.py \\
        --demo-file datasets/datasets/libero_spatial/<task>_demo.hdf5 \\
        [--action-threshold 0.005] \\
        [--min-static-run 3] \\
        [--inplace]

If --inplace is NOT passed the filtered file is saved as <name>_filtered.hdf5.
"""

import argparse
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Format detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_format(f: h5py.File) -> str:
    """Return 'libero_std' or 'intermediate'."""
    demo_keys = [k for k in f["data"].keys() if k.startswith("demo")]
    if not demo_keys:
        raise ValueError("No demo groups found under /data")
    first = f[f"data/{demo_keys[0]}"]
    if "obs" in first and "agentview_rgb" in first["obs"]:
        return "libero_std"
    return "intermediate"


# ──────────────────────────────────────────────────────────────────────────────
# Core mask computation
# ──────────────────────────────────────────────────────────────────────────────

def build_keep_mask(
    actions: np.ndarray,
    ee_states: Optional[np.ndarray],
    action_threshold: float,
    state_threshold: float,
    min_run: int,
) -> np.ndarray:
    """
    Return a boolean array of shape (T,) – True = keep this frame.

    A frame is "static" only when BOTH conditions hold:
      1. ||action[t, :6]||₂  <  action_threshold   (robot commanded to hold still)
      2. ||ee_states[t] - ee_states[t-1]||₂  <  state_threshold  (robot not moving)

    Filtering on action alone is insufficient: if action≈0 but the robot is
    still decelerating (state still changing), that frame carries meaningful
    training signal and should be kept.

    Parameters
    ----------
    actions          : (T, 7)  – EEF delta + gripper command.
    ee_states        : (T, 6) or None  – EEF position + axis-angle orientation.
                       If None, only the action condition is used.
    action_threshold : ||action[:6]||₂ below this → action is static.
    state_threshold  : ||Δee_state||₂ below this → robot is not moving.
    min_run          : minimum consecutive static frames before removal fires.
    """
    T = len(actions)
    action_mag = np.linalg.norm(actions[:, :6], axis=1)
    is_action_static = action_mag < action_threshold

    if ee_states is not None:
        # Δstate[0] is undefined; treat frame 0 as non-static (always kept anyway)
        delta = np.zeros(T, dtype=np.float64)
        delta[1:] = np.linalg.norm(np.diff(ee_states, axis=0), axis=1)
        is_state_static = delta < state_threshold
        is_static = is_action_static & is_state_static
    else:
        is_static = is_action_static

    keep = np.ones(T, dtype=bool)

    i = 0
    while i < T:
        if is_static[i]:
            j = i
            while j < T and is_static[j]:
                j += 1
            if (j - i) >= min_run:
                # keep first frame of run; remove the rest (except episode-last)
                for k in range(i + 1, j):
                    if k < T - 1:
                        keep[k] = False
            i = j
        else:
            i += 1

    keep[0]  = True   # always keep first
    keep[-1] = True   # always keep last (task completion)
    return keep


# ──────────────────────────────────────────────────────────────────────────────
# Per-demo copy helpers
# ──────────────────────────────────────────────────────────────────────────────

def copy_filtered_demo_libero_std(
    src_ep: h5py.Group,
    dst_ep: h5py.Group,
    keep: np.ndarray,
) -> None:
    """Copy all datasets in a libero_std demo group, filtered by keep mask."""
    n_kept = int(keep.sum())

    # Top-level datasets
    for ds_name in ("actions", "states", "rewards", "dones", "robot_states"):
        if ds_name in src_ep:
            dst_ep.create_dataset(ds_name, data=src_ep[ds_name][()][keep])

    # obs sub-group
    if "obs" in src_ep:
        obs_grp = dst_ep.create_group("obs")
        for obs_name in src_ep["obs"]:
            obs_grp.create_dataset(obs_name, data=src_ep["obs"][obs_name][()][keep])

    # Attributes
    for k, v in src_ep.attrs.items():
        if k == "num_samples":
            dst_ep.attrs["num_samples"] = n_kept
        else:
            dst_ep.attrs[k] = v


def copy_filtered_demo_intermediate(
    src_ep: h5py.Group,
    dst_ep: h5py.Group,
    keep: np.ndarray,
) -> None:
    """Copy states + actions in an intermediate demo group, filtered by keep mask."""
    dst_ep.create_dataset("states",  data=src_ep["states"][()][keep])
    dst_ep.create_dataset("actions", data=src_ep["actions"][()][keep])
    for k, v in src_ep.attrs.items():
        dst_ep.attrs[k] = v


# ──────────────────────────────────────────────────────────────────────────────
# Main file-level processing
# ──────────────────────────────────────────────────────────────────────────────

def process_file(
    demo_file: Path,
    action_threshold: float,
    state_threshold: float,
    min_run: int,
    inplace: bool,
) -> None:

    out_path = (
        demo_file.with_suffix(".tmp.hdf5")
        if inplace
        else demo_file.with_name(demo_file.stem + "_filtered.hdf5")
    )

    removed_total = 0
    kept_total    = 0

    with h5py.File(demo_file, "r") as src, h5py.File(out_path, "w") as dst:

        fmt = detect_format(src)
        print(f"[info] Detected format: {fmt}")
        if fmt == "intermediate":
            print(
                "[warning] Filtering the intermediate HDF5 (before create_dataset.py).\n"
                "          This will cause playback divergence warnings during create_dataset.\n"
                "          It is strongly recommended to filter the libero_std output instead."
            )

        # Copy root attrs
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        # Copy non-data top-level groups verbatim
        for key in src.keys():
            if key != "data":
                src.copy(key, dst)

        # Process data group
        data_grp = dst.create_group("data")
        for k, v in src["data"].attrs.items():
            data_grp.attrs[k] = v

        for demo_key in sorted(src["data"].keys()):
            src_ep  = src[f"data/{demo_key}"]
            actions = src_ep["actions"][()]

            # For libero_std use ee_states to detect real motion; else action-only
            ee_states = None
            if fmt == "libero_std" and "obs" in src_ep and "ee_states" in src_ep["obs"]:
                ee_states = src_ep["obs/ee_states"][()]

            keep = build_keep_mask(actions, ee_states, action_threshold, state_threshold, min_run)

            n_kept    = int(keep.sum())
            n_removed = len(keep) - n_kept
            removed_total += n_removed
            kept_total    += n_kept

            print(
                f"  {demo_key}: {len(actions)} frames  →  "
                f"{n_kept} kept, {n_removed} removed  "
                f"({n_removed / len(actions) * 100:.1f}% reduction)"
            )

            dst_ep = data_grp.create_group(demo_key)

            if fmt == "libero_std":
                copy_filtered_demo_libero_std(src_ep, dst_ep, keep)
            else:
                copy_filtered_demo_intermediate(src_ep, dst_ep, keep)

    # Finalise
    if inplace:
        demo_file.unlink()
        out_path.rename(demo_file)
        print(f"\n[done] Filtered in-place → {demo_file}")
    else:
        print(f"\n[done] Saved filtered file → {out_path}")

    total = kept_total + removed_total
    print(
        f"Summary: kept {kept_total}/{total} frames "
        f"({removed_total / total * 100:.1f}% removed)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter static frames from LIBERO demo.hdf5 files.\n"
            "Supports both libero_std (create_dataset output) and\n"
            "intermediate (collect_demonstration output) formats.\n"
            "Always prefer filtering the libero_std output."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--demo-file",
        required=True,
        help="Path to the demo.hdf5 to filter (libero_std or intermediate format).",
    )
    parser.add_argument(
        "--action-threshold",
        type=float,
        default=0.005,
        help=(
            "Frames with ||action[:6]||₂ below this value satisfy the action condition "
            "(default: 0.005)."
        ),
    )
    parser.add_argument(
        "--state-threshold",
        type=float,
        default=0.002,
        help=(
            "Frames where ||Δee_state||₂ (EEF position+orientation change) is below "
            "this value satisfy the state condition.  A frame is only removed when "
            "BOTH action and state conditions are met (default: 0.002)."
        ),
    )
    parser.add_argument(
        "--min-static-run",
        type=int,
        default=3,
        help=(
            "Minimum consecutive static frames required to trigger removal "
            "(default: 3)."
        ),
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the original file.  Otherwise saves as <name>_filtered.hdf5.",
    )

    args = parser.parse_args()
    demo_path = Path(args.demo_file)
    if not demo_path.exists():
        print(f"[error] File not found: {demo_path}")
        raise SystemExit(1)

    print(f"[info] Processing      : {demo_path}")
    print(f"[info] Action threshold: {args.action_threshold}")
    print(f"[info] State threshold : {args.state_threshold}")
    print(f"[info] Min run         : {args.min_static_run}")
    print(f"[info] In-place        : {args.inplace}")
    print()

    process_file(demo_path, args.action_threshold, args.state_threshold, args.min_static_run, args.inplace)


if __name__ == "__main__":
    main()
