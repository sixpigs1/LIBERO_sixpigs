# LIBERO 代码库分析与数据收集完整指南

> 文档撰写日期：2026-04-14
> 仓库：`LIBERO_sixpigs`（基于 [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)）

---

## 目录

1. [整体代码库结构分析](#1-整体代码库结构分析)
2. [自制数据集的完整流程](#2-自制数据集的完整流程)
3. [关键脚本深度解析](#3-关键脚本深度解析)
   - [collect_demonstration.py](#31-collect_demonstrationpy)
   - [create_dataset.py](#32-create_datasetpy)

---

## 1. 整体代码库结构分析

### 1.1 顶层目录

```
LIBERO_sixpigs/
├── libero/                  # 核心代码包（双层结构）
│   ├── libero/              # 环境、任务定义、Benchmark 注册
│   ├── lifelong/            # 终身学习 / 训练 / 评测主逻辑
│   └── configs/             # Hydra 配置文件（YAML）
├── scripts/                 # 数据收集与处理脚本
├── benchmark_scripts/       # 下载数据集、检查任务等辅助脚本
├── notebooks/               # Jupyter 教程
├── templates/               # 自定义任务模板
├── requirements.txt
└── setup.py
```

---

### 1.2 `libero/libero/` —— 仿真环境与任务定义层

这是 LIBERO 的"底层引擎"，负责构建 MuJoCo 仿真环境、解析任务描述、注册 Benchmark。

| 子模块                               | 作用                                                                                                                                                                                                       |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `envs/`                              | 所有仿真环境类。`bddl_base_domain.py` 维护 `TASK_MAPPING`（任务名→环境类的映射）；`env_wrapper.py` 提供 `OffScreenRenderEnv`（离屏渲染，用于训练）和 `SegmentationRenderEnv`；`venv.py` 提供向量化并行环境 |
| `envs/bddl_utils.py`                 | 解析 `.bddl` 任务描述文件，提取 `problem_name`、`domain_name`、`language_instruction`                                                                                                                      |
| `bddl_files/`                        | 所有任务的 `.bddl` 描述文件，按任务套件分文件夹存放（共 5 套：`libero_spatial`、`libero_object`、`libero_goal`、`libero_90`、`libero_10`）                                                                 |
| `benchmark/__init__.py`              | 注册所有 `Benchmark` 类（`LIBERO_SPATIAL`、`LIBERO_OBJECT`、`LIBERO_GOAL`、`LIBERO_90`、`LIBERO_10`、`LIBERO_100`）；定义 `Task` 数据结构和任务顺序                                                        |
| `benchmark/libero_suite_task_map.py` | 维护每个套件中所有任务名称的列表（任务名与 `.bddl` 文件名一一对应）                                                                                                                                        |
| `init_files/`                        | 每个任务的初始状态文件（`.pruned_init`），用于评测时固定初始状态、保证可复现性                                                                                                                             |
| `assets/`                            | MuJoCo XML 场景文件、物体模型、纹理贴图等仿真资源                                                                                                                                                          |
| `utils/`                             | 通用工具函数（如 `postprocess_model_xml`）                                                                                                                                                                 |

---

### 1.3 `libero/lifelong/` —— 训练与评测层

这是 LIBERO 的"上层算法"，负责终身学习的训练循环、策略网络、数据加载。

| 子模块        | 作用                                                                                                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `main.py`     | 主入口，使用 **Hydra** 读取配置，编排整个训练流程                                                                                                                        |
| `algos/`      | 终身学习算法实现：`base.py`（顺序微调）、`er.py`（经验回放）、`ewc.py`（EWC）、`packnet.py`（PackNet）、`multitask.py`（多任务联合训练）、`single_task.py`（单任务基线） |
| `models/`     | 策略网络：`bc_rnn_policy.py`（BC+RNN）、`bc_transformer_policy.py`（BC+Transformer）、`bc_vilt_policy.py`（BC+ViLT，视觉-语言联合）                                      |
| `datasets.py` | 基于 **Robomimic** 的 `SequenceDataset` 封装，将 HDF5 演示数据加载为序列训练集；`SequenceVLDataset` 在观测基础上附加语言嵌入                                             |
| `evaluate.py` | 在仿真环境中执行策略并计算成功率                                                                                                                                         |
| `metric.py`   | 终身学习指标计算（前向迁移、遗忘等）                                                                                                                                     |
| `utils.py`    | 工具函数：控制随机种子、获取任务嵌入（BERT/One-hot）、创建实验目录等                                                                                                     |

---

### 1.4 `libero/configs/` —— Hydra 配置系统

采用 **Hydra** 进行分层配置管理，`config.yaml` 是顶层配置，通过 `defaults` 组合子配置：

```
configs/
├── config.yaml              # 顶层入口（指定 benchmark、seed、device 等）
├── data/default.yaml        # 数据加载参数（obs模态、seq_len等）
├── policy/                  # 各策略网络的超参数
├── lifelong/                # 各算法的超参数
├── train/default.yaml       # 训练超参数（batch_size、lr、epoch等）
└── eval/default.yaml        # 评测参数
```

---

### 1.5 `scripts/` —— 数据生产脚本

| 脚本                            | 作用                                                                               |
| ------------------------------- | ---------------------------------------------------------------------------------- |
| `collect_demonstration.py`      | **第一步**：人工遥控操作（键盘/SpaceMouse）收集原始演示轨迹，存为 HDF5（中间格式） |
| `create_dataset.py`             | **第二步**：回放原始演示、渲染图像、提取观测，生成训练用的标准格式 HDF5            |
| `create_libero_task_example.py` | 演示如何通过程序化方式创建新任务                                                   |
| `get_dataset_info.py`           | 打印 HDF5 数据集的统计信息                                                         |
| `check_dataset_integrity.py`    | 校验数据集完整性                                                                   |

---

### 1.6 `benchmark_scripts/` —— 数据集下载与验证

| 脚本                          | 作用                                            |
| ----------------------------- | ----------------------------------------------- |
| `download_libero_datasets.py` | 从官方或 HuggingFace 下载 LIBERO 官方演示数据集 |
| `check_task_suites.py`        | 检查任务套件是否完整                            |
| `render_single_task.py`       | 渲染单个任务的演示视频                          |

---

### 1.7 任务体系总览

LIBERO 共有 **130 个任务**，分为 5 个套件：

| 套件             | 任务数 | 核心考察能力                       | 典型任务示例                 |
| ---------------- | ------ | ---------------------------------- | ---------------------------- |
| `libero_spatial` | 10     | 空间位置理解（同一物体，不同位置） | "把盘子旁边的黑碗放到盘子上" |
| `libero_object`  | 10     | 物体识别（不同物体，相同动作）     | "把番茄酱放进篮子"           |
| `libero_goal`    | 10     | 目标泛化（相同场景，不同目标）     | "打开柜子中间的抽屉"         |
| `libero_90`      | 90     | 预训练（知识纠缠，多样场景）       | 多种厨房/客厅/书房场景操作   |
| `libero_10`      | 10     | 下游终身学习测试                   | 复合操作任务                 |

---

## 2. 自制数据集的完整流程

### 前提条件

确保已安装所有依赖，并激活对应 conda 环境：

```bash
conda activate libero
```

---

### Step 1：确认目标任务的 BDDL 文件

LIBERO 的每个任务由一个 `.bddl` 文件定义，位于：

```
libero/libero/bddl_files/<suite_name>/<task_name>.bddl
```

例如，`libero_spatial` 套件的第一个任务：

```
libero/libero/bddl_files/libero_spatial/
    pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl
```

**如何列出某套件所有任务名称：**

```python
from libero.libero.benchmark import get_benchmark_dict
benchmark = get_benchmark_dict()["libero_spatial"]()
for i, task in enumerate(benchmark.tasks):
    print(f"[{i}] {task.name}")
    print(f"     语言指令: {task.language}")
    print(f"     BDDL文件: {task.bddl_file}")
```

---

### Step 2：收集人工遥控演示（`collect_demonstration.py`）

此脚本打开一个 **MuJoCo 可视化窗口**，由人通过键盘或 SpaceMouse 遥控机械臂完成任务，实时录制动作序列。

**最小运行命令（键盘控制）：**

```bash
cd scripts
python collect_demonstration.py \
    --bddl-file ../libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl \
    --device keyboard \
    --num-demonstration 50 \
    --directory ./demonstration_data
```

**SpaceMouse 控制命令（推荐，操作更流畅）：**

```bash
cd scripts
python collect_demonstration.py \
    --bddl-file ../libero/libero/bddl_files/libero_spatial/<task_name>.bddl \
    --device spacemouse \
    --vendor-id 9583 \
    --product-id 50734 \
    --num-demonstration 50 \
    --directory ./demonstration_data
```

**收集完成后的输出目录结构：**

```
demonstration_data/
└── robosuite_ln_<ProblemName>_<timestamp>_<language>/
    ├── demo.hdf5            # 汇总后的中间格式HDF5（含states和actions）
    └── (tmp/ 临时目录已被内部清理)
```

---

### Step 3：生成训练格式数据集（`create_dataset.py`）

此脚本读取 Step 2 产出的 `demo.hdf5`，在仿真中**回放**每条轨迹，渲染图像并提取观测，生成训练用的标准 HDF5。

**运行命令（推荐启用摄像头观测）：**

```bash
cd scripts
python create_dataset.py \
    --demo-file ./demonstration_data/robosuite_ln_<ProblemName>_<timestamp>_<language>/demo.hdf5 \
    --use-camera-obs \
    --use-actions
```

**如需深度图像：**

```bash
python create_dataset.py \
    --demo-file ./demonstration_data/.../demo.hdf5 \
    --use-camera-obs \
    --use-depth \
    --use-actions
```

**输出位置：** 脚本会自动根据 BDDL 文件路径，将结果写入 LIBERO 标准数据集目录：

```
<libero_datasets_path>/<suite_name>/<task_name>_demo.hdf5
```

例如：`~/.datasets/libero_spatial/pick_up_the_black_bowl_..._demo.hdf5`

---

### Step 4：验证数据集

```bash
cd scripts
python get_dataset_info.py --dataset-path <上一步生成的_demo.hdf5路径>
python check_dataset_integrity.py
```

---

### Step 5（可选）：使用数据集训练策略

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python libero/lifelong/main.py \
    seed=10000 \
    benchmark_name=LIBERO_SPATIAL \
    policy=bc_transformer_policy \
    lifelong=base
```

---

### 完整流程一览

```
[BDDL 文件]
     │
     ▼
collect_demonstration.py  ──(人工遥控)──►  demo.hdf5（中间格式，含states+actions）
     │
     ▼
create_dataset.py  ──(仿真回放+渲染)──►  <task_name>_demo.hdf5（标准训练格式，含图像+本体感知）
     │
     ▼
lifelong/main.py  ──(策略训练)──►  训练好的模型
```

---

## 3. 关键脚本深度解析

---

### 3.1 `collect_demonstration.py`

#### 作用与功能

**功能定位**：第一阶段数据收集入口。打开 MuJoCo 渲染窗口，允许操作者通过输入设备（键盘/SpaceMouse）实时遥控 Panda 机械臂完成指定任务，并将成功的演示轨迹保存为 HDF5 文件。

**核心工作流程**：
1. 加载 BDDL 文件，解析任务名、场景、语言指令
2. 创建带有 `DataCollectionWrapper` 包裹的仿真环境（负责自动录制每一步的 state 和 action）
3. 初始化输入设备（键盘或 SpaceMouse）
4. 循环收集：每次人工操作一条轨迹 → 若任务成功则保留，失败则丢弃
5. 调用 `gather_demonstrations_as_hdf5()` 将所有 NPZ 临时文件汇总写入 `demo.hdf5`

**HDF5 中间文件结构**：
```
demo.hdf5
└── data/                         # 顶层 group
    ├── (attrs) date, time        # 元数据
    ├── (attrs) env_info          # 环境配置 JSON
    ├── (attrs) bddl_file_name    # BDDL 文件路径
    ├── (attrs) bddl_file_content # BDDL 文件内容
    ├── (attrs) problem_info      # 任务信息（problem_name, language_instruction等）
    ├── demo_1/
    │   ├── (attrs) model_file    # 该条演示的 MuJoCo XML 场景描述
    │   ├── states                # MuJoCo 物理状态序列 (T, state_dim)
    │   └── actions               # 控制指令序列 (T, 7)：[dx,dy,dz,droll,dpitch,dyaw,gripper]
    ├── demo_2/
    └── ...
```

---

#### 参数详解

| 参数                  | 类型       | 默认值               | 作用                                                                          | 填写建议                                                                       |
| --------------------- | ---------- | -------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `--bddl-file`         | `str`      | **必填**             | 指定任务的 `.bddl` 文件绝对/相对路径，决定了要收集哪个任务的数据              | 填写 `libero/libero/bddl_files/<suite>/<task>.bddl` 的完整路径                 |
| `--num-demonstration` | `int`      | `50`                 | 要收集的成功演示条数，到达此数量后脚本自动退出                                | LIBERO 官方数据集每个任务有 50 条，建议设为 **50**；快速测试可用 **10**        |
| `--directory`         | `str`      | `demonstration_data` | 中间原始数据和最终 `demo.hdf5` 的存放根目录                                   | 建议使用有意义的路径，如 `./my_demos`                                          |
| `--device`            | `str`      | `spacemouse`         | 输入设备类型：`keyboard`（键盘）或 `spacemouse`（3D 鼠标）                    | 没有 SpaceMouse 设备时填 `keyboard`；有条件强烈推荐 `spacemouse`，操作精度更高 |
| `--robots`            | `str/list` | `Panda`              | 使用的机器人型号，支持 Robosuite 中所有机器人（如 `Panda`、`Sawyer`、`IIWA`） | 对于 LIBERO 官方任务，保持默认 **`Panda`**                                     |
| `--controller`        | `str`      | `OSC_POSE`           | 控制器类型：`OSC_POSE`（操作空间控制，更直觉）或 `IK_POSE`（逆运动学控制）    | 保持默认 **`OSC_POSE`**，操作更流畅自然                                        |
| `--arm`               | `str`      | `right`              | 双臂机器人时指定控制哪只臂（`right`/`left`），单臂机器人忽略此参数            | 单臂 Panda 保持默认 **`right`**                                                |
| `--camera`            | `str`      | `agentview`          | 遥控时显示的摄像机视角（仅影响可视化显示，不影响录制内容）                    | 默认 `agentview` 即可；可选 `frontview`、`sideview`                            |
| `--config`            | `str`      | `single-arm-opposed` | 双臂机器人的手臂排列配置，单臂机器人忽略此参数                                | 使用单臂 Panda 时保持默认                                                      |
| `--pos-sensitivity`   | `float`    | `1.5`                | 位置输入的灵敏度缩放系数，值越大机械臂移动越快                                | 根据个人操作习惯调整，建议范围 `1.0 ~ 3.0`；初学者可降低到 `1.0`               |
| `--rot-sensitivity`   | `float`    | `1.0`                | 旋转输入的灵敏度缩放系数                                                      | 同上，建议范围 `0.5 ~ 2.0`                                                     |
| `--vendor-id`         | `int`      | `9583`               | SpaceMouse 的 USB Vendor ID（设备识别用），仅当 `--device spacemouse` 时有效  | 3Dconnexion SpaceMouse 默认为 `9583`，不同型号可能不同，可通过 `lsusb` 查看    |
| `--product-id`        | `int`      | `50734`              | SpaceMouse 的 USB Product ID，仅当 `--device spacemouse` 时有效               | 默认 `50734`（SpaceMouse Compact），不同型号需查 `lsusb`                       |

#### 操作说明（键盘控制时）

| 键位        | 动作                       |
| ----------- | -------------------------- |
| `W/S`       | 机械臂末端前进/后退（X轴） |
| `A/D`       | 机械臂末端左右移动（Y轴）  |
| `Q/E`       | 机械臂末端上下移动（Z轴）  |
| `Z/X`       | 末端旋转                   |
| 空格键      | 切换夹爪开/闭              |
| `Q`（退出） | 放弃当前演示（不保存）     |

---

### 3.2 `create_dataset.py`

#### 作用与功能

**功能定位**：第二阶段数据处理与格式转换。读取 `collect_demonstration.py` 产出的中间 HDF5，在仿真中**逐帧回放**每条轨迹，同时渲染摄像头图像、提取机器人本体感知状态，生成可供策略网络直接训练的标准格式 HDF5。

**核心工作流程**：
1. 读取 `demo.hdf5`，解析任务环境信息和 BDDL 路径
2. 重建仿真环境（自动配置双摄像头：`agentview` + `robot0_eye_in_hand`，分辨率 128×128）
3. 逐条 Demo 回放：
   - 从保存的 XML 重置场景，精确还原初始状态
   - 重放 actions，每步提取 obs（图像/关节角/末端位姿/夹爪状态）
   - 跳过前 5 帧（`cap_index=5`，避免力传感器初始不稳定造成的噪声）
   - 检查回放误差，若偏差 > 0.01 会打印警告
4. 写入标准格式 HDF5（含完整 obs、actions、rewards、dones、robot_states）

**输出 HDF5 标准格式结构**：
```
<task_name>_demo.hdf5
└── data/
    ├── (attrs) env_name, problem_info, bddl_file_name, env_args
    ├── demo_0/
    │   ├── (attrs) num_samples, model_file, init_state
    │   ├── obs/
    │   │   ├── agentview_rgb         # (T, H, W, 3) uint8，第三人称视角 RGB
    │   │   ├── eye_in_hand_rgb       # (T, H, W, 3) uint8，腕部摄像头 RGB
    │   │   ├── gripper_states        # (T, 2) float，夹爪关节角
    │   │   ├── joint_states          # (T, 7) float，关节角度
    │   │   ├── ee_states             # (T, 6) float，末端位置+轴角旋转
    │   │   ├── ee_pos                # (T, 3) float，末端 XYZ 位置
    │   │   └── ee_ori                # (T, 3) float，末端轴角旋转
    │   ├── actions                   # (T, 7) float，控制动作
    │   ├── states                    # (T, state_dim) float，MuJoCo 状态
    │   ├── robot_states              # (T, robot_state_dim) float，机器人状态向量
    │   ├── rewards                   # (T,) uint8，稀疏奖励（仅最后一步为1）
    │   └── dones                     # (T,) uint8，终止标志（仅最后一步为1）
    └── demo_1/ ...
```

---

#### 参数详解

| 参数               | 类型   | 默认值                        | 作用                                                                                                                                                                   | 填写建议                                                    |
| ------------------ | ------ | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `--demo-file`      | `str`  | `demo.hdf5`                   | 输入文件：`collect_demonstration.py` 产出的中间 HDF5 路径                                                                                                              | 填写 Step 2 生成的 `demo.hdf5` 的完整路径，**必须正确填写** |
| `--use-camera-obs` | `flag` | `False`（未启用）             | **是否渲染并保存摄像头图像**（`agentview_rgb` 和 `eye_in_hand_rgb`）。不加此参数则只保存本体感知（低维状态），**基于视觉的策略（如 BC-Transformer、BC-ViLT）必须开启** | 几乎所有情况下都应加上 `--use-camera-obs`                   |
| `--use-actions`    | `flag` | `False`（未启用）             | 是否从原始 HDF5 直接使用录制的 actions（而非从物理状态中重新计算）。**强烈建议开启**                                                                                   | 始终加上 `--use-actions`，保证动作与原始演示完全一致        |
| `--use-depth`      | `flag` | `False`（未启用）             | 是否额外渲染并保存深度图像（`agentview_depth` 和 `eye_in_hand_depth`）。需要同时开启 `--use-camera-obs`                                                                | 如不需要深度信息，**不加此参数**可大幅减小数据集体积        |
| `--no-proprio`     | `flag` | `False`（即默认保存本体感知） | 是否**不保存**本体感知观测（joint_states、ee_states、gripper_states）。加上此参数则只有图像数据                                                                        | 通常**不加此参数**，保留本体感知对策略学习非常有益          |
| `--dataset-path`   | `str`  | `datasets/`                   | 输出数据集的根目录（注意：实际输出路径由 BDDL 文件路径自动决定，此参数在当前代码版本中主要作为参考）                                                                   | 保持默认或按需指定                                          |
| `--dataset-name`   | `str`  | `training_set`                | 数据集名称标识（同上，在当前代码中作为元数据标识）                                                                                                                     | 保持默认即可                                                |

---

#### 两个脚本的对比总结

| 维度                 | `collect_demonstration.py`                  | `create_dataset.py`                              |
| -------------------- | ------------------------------------------- | ------------------------------------------------ |
| **执行阶段**         | 第 1 阶段（数据采集）                       | 第 2 阶段（数据处理）                            |
| **输入**             | BDDL 文件 + 人工操作                        | 中间 `demo.hdf5`                                 |
| **输出**             | 中间格式 `demo.hdf5`（含 states + actions） | 标准训练 `<task>_demo.hdf5`（含图像 + 本体感知） |
| **仿真渲染**         | 实时有屏渲染（给人看）                      | 离屏渲染（给模型看）                             |
| **数据内容**         | 仅 MuJoCo 状态 + 原始动作                   | 完整观测（RGB图像 + 本体感知 + 动作 + 奖励）     |
| **是否需要人工干预** | 是（人工遥控操作）                          | 否（全自动回放）                                 |
| **耗时**             | 取决于演示数量和操作熟练程度                | 取决于演示数量和渲染速度（CPU/GPU）              |

---

#### 常见问题与注意事项

1. **`cap_index=5`（跳帧问题）**：`create_dataset.py` 中硬编码了跳过每条轨迹前 5 帧（`cap_index = 5`），原因是力传感器在仿真初始时不稳定。这是写死的值，若需更改需直接修改源码。

2. **摄像头分辨率**：`create_dataset.py` 中摄像头分辨率硬编码为 128×128，如需更高分辨率需修改源码中的 `camera_heights` 和 `camera_widths`。

3. **输出路径自动推断**：`create_dataset.py` 会根据 `demo.hdf5` 中记录的 `bddl_file_name` 自动推断输出路径（写入 `get_libero_path("datasets")` 下），不需要手动指定输出位置。

4. **回放误差警告**：若看到 `[warning] playback diverged by X.XX for ep ...`，说明仿真回放与原始录制存在偏差。误差 < 0.01 可忽略；误差过大说明仿真不确定性较高，该演示数据质量可能较差。

5. **批量处理多个任务**：如需对一个套件的所有任务收集数据，建议写 bash 脚本循环调用，将 `--bddl-file` 替换为每个任务的路径。

---

*文档基于 `LIBERO_sixpigs` 仓库 `master` 分支代码分析生成。*
