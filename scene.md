# LIBERO Suite → Scene XML 对应关系

## 各 Suite 使用的 Scene XML

| Suite | Problem 类名 | Scene XML 文件 | 任务数 |
|-------|-------------|----------------|--------|
| `libero_spatial` | `LIBERO_Tabletop_Manipulation` | `scenes/libero_tabletop_base_style.xml` | 10 |
| `libero_goal` | `LIBERO_Tabletop_Manipulation` | `scenes/libero_tabletop_base_style.xml` | 10 |
| `libero_object` | `LIBERO_Floor_Manipulation` | `scenes/libero_floor_base_style.xml` | 10 |
| `libero_10` | `LIBERO_Kitchen_Tabletop_Manipulation` | `scenes/libero_kitchen_tabletop_base_style.xml` | 4 |
| `libero_10` | `LIBERO_Living_Room_Tabletop_Manipulation` | `scenes/libero_living_room_tabletop_base_style.xml` | 5 |
| `libero_10` | `LIBERO_Study_Tabletop_Manipulation` | `scenes/libero_study_base_style.xml` | 1 |
| `libero_90` | `LIBERO_Kitchen_Tabletop_Manipulation` | `scenes/libero_kitchen_tabletop_base_style.xml` | 46 |
| `libero_90` | `LIBERO_Living_Room_Tabletop_Manipulation` | `scenes/libero_living_room_tabletop_base_style.xml` | 27 |
| `libero_90` | `LIBERO_Study_Tabletop_Manipulation` | `scenes/libero_study_base_style.xml` | 17 |

> 所有 XML 文件路径均相对于 `libero/libero/assets/`

---

## Problem 类 → XML 的代码映射

| Problem 类 | 源文件 | scene_xml 默认值 |
|-----------|--------|-----------------|
| `Libero_Tabletop_Manipulation` | `envs/problems/libero_tabletop_manipulation.py` | `scenes/libero_tabletop_base_style.xml` |
| `Libero_Floor_Manipulation` | `envs/problems/libero_floor_manipulation.py` | `scenes/libero_floor_base_style.xml` |
| `Libero_Kitchen_Tabletop_Manipulation` | `envs/problems/libero_kitchen_tabletop_manipulation.py` | `scenes/libero_kitchen_tabletop_base_style.xml` |
| `Libero_Living_Room_Tabletop_Manipulation` | `envs/problems/libero_living_room_tabletop_manipulation.py` | `scenes/libero_living_room_tabletop_base_style.xml` |
| `Libero_Study_Tabletop_Manipulation` | `envs/problems/libero_study_tabletop_manipulation.py` | `scenes/libero_study_base_style.xml` |
| `Libero_Coffee_Table_Manipulation` | `envs/problems/libero_coffee_table_manipulation.py` | `scenes/libero_coffee_table_base_style.xml` |

---

## sideview_rear 相机现状

`sideview_rear` 相机目前**只添加在** `libero_tabletop_base_style.xml` 中，因此：

| Suite | 是否有 `sideview_rear` |
|-------|----------------------|
| `libero_spatial` | ✅ 有 |
| `libero_goal` | ✅ 有 |
| `libero_object` | ❌ 无（使用 `libero_floor_base_style.xml`）|
| `libero_10` | ❌ 无（使用 kitchen / living_room / study XML）|
| `libero_90` | ❌ 无（使用 kitchen / living_room / study XML）|

需要在以下 4 个 XML 文件中同样添加 `sideview_rear` 相机定义（或使用其他已有相机）：

- `libero/libero/assets/scenes/libero_floor_base_style.xml`
- `libero/libero/assets/scenes/libero_kitchen_tabletop_base_style.xml`
- `libero/libero/assets/scenes/libero_living_room_tabletop_base_style.xml`
- `libero/libero/assets/scenes/libero_study_base_style.xml`
