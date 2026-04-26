# convert_libero_to_lerobot.py 使用说明

## 功能概述

将 `scripts/create_dataset.py` 产出的 LIBERO HDF5 数据集批量转换为 **LeRobot v2.1** 格式数据集。

支持将**同一任务套件下的多个任务文件**（如 `libero_spatial` 下 10 个任务的 HDF5）合并转换为一个统一的 LeRobot 数据集，不同任务通过 `task_index` 区分。

---

## 输入格式（LIBERO HDF5）

`create_dataset.py` 产出的文件，存放在以下路径：

```
datasets/datasets/<suite_name>/
    <task_name_1>_demo.hdf5
    <task_name_2>_demo.hdf5
    ...
```

每个 HDF5 文件内部结构：
```
data/
  (attrs) problem_info       ← 包含 language_instruction（任务语言描述）
  demo_0/
    actions                  (T, 7)   float64
    obs/agentview_rgb        (T, H, W, 3) uint8
    obs/eye_in_hand_rgb      (T, H, W, 3) uint8
    obs/ee_states            (T, 6)   float64
    obs/gripper_states       (T, 2)   float64
    ...
  demo_1/ ...
```

---

## 输出格式（LeRobot v2.1）

```
<output_dir>/
├── meta/
│   ├── info.json             ← 数据集入口文件（总帧数、任务数、feature 定义等）
│   ├── tasks.jsonl           ← 每行：{"task_index": N, "task": "语言指令"}
│   ├── episodes.jsonl        ← 每行：{"episode_index": N, "tasks": [...], "length": T}
│   └── episodes_stats.jsonl  ← 每行：每个 episode 各特征的 min/max/mean/std
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.agentview_rgb/
        │   ├── episode_000000.mp4
        │   └── ...
        └── observation.images.eye_in_hand_rgb/
            ├── episode_000000.mp4
            └── ...
```

### Parquet 文件列说明

| 列名                | 类型    | 维度   | 说明                                                            |
| ------------------- | ------- | ------ | --------------------------------------------------------------- |
| `observation.state` | float32 | (8,)   | `ee_states(6)` + `gripper_states(2)`，即末端位姿(xyz+轴角)+夹爪 |
| `action`            | float32 | (7,)   | OSC_POSE 控制指令(xyz+轴角+夹爪)                                |
| `timestamp`         | float32 | scalar | 该帧在 episode 内的时间戳（秒），step/FPS                       |
| `frame_index`       | int64   | scalar | 该帧在当前 episode 内的索引（从0开始）                          |
| `episode_index`     | int64   | scalar | 该帧所属 episode 的全局编号                                     |
| `index`             | int64   | scalar | 该帧在整个数据集中的全局索引                                    |
| `task_index`        | int64   | scalar | 对应 `tasks.jsonl` 中的任务编号                                 |

---

## 运行环境要求

脚本**不依赖** libero conda 环境，仅需以下常见 Python 包：

```bash
pip install numpy opencv-python h5py pyarrow
```

---

## 使用方法

### 基本命令

```bash
python scripts/convert_libero_to_lerobot.py \
    --input-dir  <HDF5文件所在目录> \
    --output-dir <LeRobot数据集输出目录>
```

### 参数说明

| 参数           | 是否必填 | 默认值   | 说明                                                                               |
| -------------- | -------- | -------- | ---------------------------------------------------------------------------------- |
| `--input-dir`  | **必填** | —        | 包含一个或多个 `*_demo.hdf5` 文件的目录，对应同一个任务套件（如 `libero_spatial`） |
| `--output-dir` | **必填** | —        | LeRobot 格式数据集的根输出目录，不存在会自动创建                                   |
| `--robot-type` | 可选     | `franka` | 写入 `info.json` 的机器人类型标识                                                  |

---

## 使用示例

### 示例 1：转换 libero_spatial 套件

```bash
python scripts/convert_libero_to_lerobot.py \
    --input-dir  datasets/datasets/libero_spatial \
    --output-dir datasets/lerobot/libero_spatial_lerobot
```

### 示例 2：转换 libero_10 套件（多任务合并）

```bash
python scripts/convert_libero_to_lerobot.py \
    --input-dir  datasets/datasets/libero_10 \
    --output-dir datasets/lerobot/libero_10_lerobot
```

转换后，`meta/tasks.jsonl` 中会自动记录所有 10 个任务的语言描述，每条 episode 通过 `task_index` 字段关联到对应任务。

---

## 注意事项

1. **图像分辨率**：转换时直接使用 HDF5 中存储的原始分辨率（默认 128×128，可在 `create_dataset.py` 中调整 `camera_heights`/`camera_widths` 以获取更高分辨率），输出的 `info.json` 会自动匹配实际分辨率。

2. **多任务合并规则**：`--input-dir` 下的所有 `*_demo.hdf5` 文件都会被整合。每个 HDF5 文件对应一个任务（通过文件内的 `language_instruction` 区分），`task_index` 按文件名排序自动分配。

3. **chunks 分片**：每个 chunk 最多包含 1000 条 episodes（`CHUNKS_SIZE = 1000`），超过时自动创建 `chunk-001` 等子目录，符合 LeRobot 标准。

4. **视频编码**：使用 OpenCV 的 `mp4v` codec 编码，生成标准 `.mp4` 文件。

5. **`index` 全局连续性**：parquet 中的 `index` 列是跨所有 episodes 的全局帧索引，在整个数据集内单调递增，LeRobot 的 `SequenceDataset` 依赖这一特性来做连续采样。

---

## 完整数据生产流程

```
1. [人工操作] collect_demonstration.py  → demo.hdf5（中间格式）
        ↓
2. [仿真回放] create_dataset.py          → <task>_demo.hdf5（LIBERO 标准格式）
        ↓
3. [格式转换] convert_libero_to_lerobot.py → LeRobot v2.1 数据集
```
