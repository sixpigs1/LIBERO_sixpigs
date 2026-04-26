"""
inspect_hdf5.py — 查看 LIBERO 数据流水线中各阶段 HDF5 文件的内容

支持三种格式：
  1. collected_demos/demo.hdf5        ← collect_demonstration.py 产出（中间格式）
  2. datasets/.../xxx_demo.hdf5       ← create_dataset.py 产出（LIBERO 标准格式）
  3. 自动识别格式

用法：
  python scripts/inspect_hdf5.py <path_to_hdf5> [--demo DEMO_KEY] [--show-data]
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# 格式自动识别
# ──────────────────────────────────────────────────────────────────────────────

def detect_format(f: h5py.File) -> str:
    """
    返回：
      'intermediate'  → collect_demonstration.py 产出
      'libero_std'    → create_dataset.py 产出
      'unknown'
    """
    keys = list(f.keys())
    if "data" not in keys:
        return "unknown"
    data_keys = list(f["data"].keys())
    if not data_keys:
        return "unknown"
    first_demo = f["data"][data_keys[0]]
    demo_sub = list(first_demo.keys())
    if "obs" in demo_sub:
        return "libero_std"
    # intermediate 格式 demo 下只有 actions, states
    if set(demo_sub) <= {"actions", "states"}:
        return "intermediate"
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# 通用工具
# ──────────────────────────────────────────────────────────────────────────────

def arr_summary(arr: np.ndarray) -> str:
    """返回 ndarray 的简洁统计描述。"""
    return (
        f"shape={arr.shape}  dtype={arr.dtype}  "
        f"min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}"
    )


def print_section(title: str):
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subsection(title: str):
    print(f"\n  ── {title} ──")


# ──────────────────────────────────────────────────────────────────────────────
# 格式 1：中间格式（collect_demonstration.py 产出）
# ──────────────────────────────────────────────────────────────────────────────

def inspect_intermediate(f: h5py.File, demo_key: Optional[str], show_data: bool):
    print_section("格式：中间格式  (collect_demonstration.py 产出)")

    data = f["data"]
    demo_keys = sorted(data.keys())

    # ── 全局元数据 ────────────────────────────────────────────────────────────
    print_subsection("全局属性 (data.attrs)")
    for k, v in data.attrs.items():
        val_str = str(v)
        if k == "problem_info":
            try:
                val_str = json.dumps(json.loads(v), ensure_ascii=False)
            except Exception:
                pass
        if k == "bddl_file_content":
            val_str = val_str[:80] + "..."
        elif k == "env_info":
            val_str = val_str[:100] + "..."
        print(f"    {k:30s}: {val_str}")

    # ── 概览 ──────────────────────────────────────────────────────────────────
    print_subsection("Episode 概览")
    print(f"    共 {len(demo_keys)} 条演示：{demo_keys}")
    for dk in demo_keys:
        ep = data[dk]
        T = ep["states"].shape[0]
        print(f"    {dk}  →  T={T} 步")

    # ── 指定 demo 详情 ────────────────────────────────────────────────────────
    if demo_key is None:
        demo_key = demo_keys[0]
        print(f"\n    (未指定 --demo，默认展示 {demo_key})")

    if demo_key not in data:
        print(f"[error] demo '{demo_key}' 不存在，可选：{demo_keys}")
        return

    ep = data[demo_key]
    print_subsection(f"Demo: {demo_key}  (attrs)")
    for k, v in ep.attrs.items():
        val_str = str(v)
        if k == "model_file":
            val_str = val_str[:80] + "..."
        print(f"    {k:20s}: {val_str}")

    print_subsection(f"Demo: {demo_key}  (datasets)")
    for k in ep.keys():
        arr = ep[k][()]
        print(f"    {k:20s}: {arr_summary(arr)}")

    if show_data:
        print_subsection(f"Demo: {demo_key}  (前3行数据预览)")
        for k in ep.keys():
            if k in ("actions"):
                arr = ep[k][()]
                print(f"    {k}[0:30]:")
                print(f"      {arr[:80]}")


# ──────────────────────────────────────────────────────────────────────────────
# 格式 2：LIBERO 标准格式（create_dataset.py 产出）
# ──────────────────────────────────────────────────────────────────────────────

def inspect_libero_std(f: h5py.File, demo_key: Optional[str], show_data: bool):
    print_section("格式：LIBERO 标准格式  (create_dataset.py 产出)")

    data = f["data"]
    demo_keys = sorted(data.keys())

    # ── 全局元数据 ────────────────────────────────────────────────────────────
    print_subsection("全局属性 (data.attrs)")
    for k, v in data.attrs.items():
        val_str = str(v)
        if k in ("problem_info", "env_args"):
            try:
                val_str = json.dumps(json.loads(v), ensure_ascii=False, indent=6)
            except Exception:
                pass
        if k in ("bddl_file_content",):
            val_str = val_str[:80] + "..."
        print(f"    {k:30s}: {val_str}")

    # ── 概览 ──────────────────────────────────────────────────────────────────
    print_subsection("Episode 概览")
    total_frames = sum(data[dk]["actions"].shape[0] for dk in demo_keys)
    print(f"    共 {len(demo_keys)} 条演示，合计 {total_frames} 帧")
    for dk in demo_keys:
        ep = data[dk]
        T = ep["actions"].shape[0]
        obs_keys = sorted(ep["obs"].keys())
        print(f"    {dk}  →  T={T} 步  |  obs keys: {obs_keys}")

    # ── 指定 demo 详情 ────────────────────────────────────────────────────────
    if demo_key is None:
        demo_key = demo_keys[0]
        print(f"\n    (未指定 --demo，默认展示 {demo_key})")

    if demo_key not in data:
        print(f"[error] demo '{demo_key}' 不存在，可选：{demo_keys}")
        return

    ep = data[demo_key]

    print_subsection(f"Demo: {demo_key}  (attrs)")
    for k, v in ep.attrs.items():
        val_str = str(v)
        if k == "model_file":
            val_str = val_str[:80] + "..."
        elif k == "init_state":
            val_str = f"array shape={np.array(v).shape}"
        print(f"    {k:20s}: {val_str}")

    print_subsection(f"Demo: {demo_key}  (顶层 datasets)")
    top_ds = [k for k in ep.keys() if k != "obs"]
    for k in top_ds:
        arr = ep[k][()]
        print(f"    {k:20s}: {arr_summary(arr)}")

    print_subsection(f"Demo: {demo_key}  (obs/* datasets)")
    for k in sorted(ep["obs"].keys()):
        arr = ep["obs"][k][()]
        print(f"    obs/{k:20s}: {arr_summary(arr)}")

    if show_data:
        print_subsection(f"Demo: {demo_key}  (前3帧数据预览)")
        for k in top_ds:
            arr = ep[k][()]
            print(f"    {k}[0:3]:\n      {arr[:3]}")
        for k in sorted(ep["obs"].keys()):
            arr = ep["obs"][k][()]
            if "rgb" in k or "depth" in k:
                print(f"    obs/{k}[0]: shape={arr[0].shape}  (图像，不打印像素值)")
            else:
                print(f"    obs/{k}[0:3]:\n      {arr[:3]}")


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="查看 LIBERO 数据流水线中各阶段 HDF5 文件的内容"
    )
    parser.add_argument(
        "hdf5_path",
        type=str,
        help="HDF5 文件路径（支持中间格式和 LIBERO 标准格式）",
    )
    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        help="指定要查看的 demo key，如 demo_0 或 demo_1（默认展示第一个）",
    )
    parser.add_argument(
        "--show-data",
        action="store_true",
        help="额外打印每个字段的前3行数值（图像跳过）",
    )
    args = parser.parse_args()

    path = Path(args.hdf5_path)
    if not path.exists():
        print(f"[error] 文件不存在：{path}")
        sys.exit(1)

    with h5py.File(path, "r") as f:
        fmt = detect_format(f)
        print(f"\n文件：{path}")
        print(f"自动识别格式：{fmt}")

        if fmt == "intermediate":
            inspect_intermediate(f, args.demo, args.show_data)
        elif fmt == "libero_std":
            inspect_libero_std(f, args.demo, args.show_data)
        else:
            print("[warning] 未能识别格式，打印原始结构：")
            def print_tree(name, obj):
                indent = "  " * name.count("/")
                if isinstance(obj, h5py.Dataset):
                    print(f"  {indent}{name}: shape={obj.shape} dtype={obj.dtype}")
                else:
                    print(f"  {indent}{name}/")
            f.visititems(print_tree)

    print()


if __name__ == "__main__":
    main()
