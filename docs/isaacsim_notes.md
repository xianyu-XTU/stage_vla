# Isaac Sim 官方文档学习笔记

> 来源：https://docs.isaacsim.omniverse.nvidia.com/latest/ 关键章节（Python 脚本 / 传感器 / 机器人仿真 / 快速上手 / 排障），2026-08-15 抓取精读。
> ⚠️ **版本提示**：文档为 /latest/，多处使用 `isaacsim.core.experimental.*` / `isaacsim.sensors.experimental.rtx`（新 API）。本地 Isaac Sim 是 **6.0.1 独立版**、Isaac Lab 3.0.0。**移植 snippet 前先确认本地包名存在**（`isaacsim.core.api.*` 还是 `experimental.*`）。本项目主要工作走 Isaac Lab 高层框架；这些原生 API 笔记用于需要下沉到 Isaac Sim 的场景（自定义传感器、调试、数据生成）。

## 一、Python 脚本化（SimulationApp 与场景搭建）

### SimulationApp 启动（standalone Python）
```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})   # headless 放 config
simulation_app.update()
simulation_app.close()
```
- **import 顺序铁律**：`SimulationApp(...)` 必须最先实例化，之后才能 import `omni.*`/`pxr.*`/`isaacsim.core.*`（API 由扩展系统在启动时提供）。
- standalone（命令行 `python.bat xxx.py`）适合自动化/批量；interactive（Script Editor）适合探索 API。
- SimulationApp = Kit 应用生命周期的高层封装（底层 carbonite/Kit）。追加扩展：`.kit` 文件 `[dependencies]` 或 `enable_extension(...)`。

### 场景不会自动"能物理"（最易踩的坑）
必须显式：`UsdPhysics.Scene.Define(stage, "/World/physicsScene")` + 设重力 + 地面 + 给对象 `RigidBodyAPI`/`CollisionAPI`（或 Core 的 `RigidPrim`/`GeomPrim`）。Isaac Lab 里这些全是默认帮你做的。

### 机器人：add_reference + Articulation 正则批处理
```python
usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
stage_utils.add_reference_to_stage(usd_path, path="/World/Franka_1", variants=[("Gripper","AlternateFinger")])
articulation = Articulation("/World/Franka_.*", positions=[[-1,-1,0],[1,1,0]])
articulation.set_dof_position_targets(0.04, dof_indices=articulation.get_dof_indices("panda_finger_joint1"))
articulation.switch_dof_control_mode("velocity")   # position(默认)/velocity/effort 三选一
```
- 关节控制接口一律**弧度**；指令数组顺序/长度必须与关节索引一一匹配（Franka = 7 臂 + 2 指 = 9 关节，手指索引 7、8）。
- **tensor 物理查询必须等 timeline 播放**（`app_utils.play()` + `await app_utils.update_app_async(steps=N)`），否则 `is_physics_tensor_entity_valid()` 为假。
- 射线/重叠查询：`omni.physx.get_physx_scene_query_interface().raycast_closest(...)`。

### 原生 vs Isaac Lab 差异
- 原生：USD prim + pxr + Core 封装，手写 PhysicsScene/地面/碰撞/材质；按 prim path（支持正则）操作。
- Isaac Lab：高层 RL 环境（`isaaclab.envs`/`isaaclab.assets.Articulation`/`SimulationContext`），按 env 索引 tensor 批处理。**`PhysicsSimStep` 是 Isaac Lab 概念，原生没有**（原生步进 = `app.update()` / timeline 播放）。

## 二、传感器 / 相机（面向 VLA 感知）

### 版本迁移（重要）
`isaacsim.sensors.camera` 自 **6.0 起已废弃** → 用 `isaacsim.sensors.experimental.rtx`。两层设计：
- **RtxCamera（authoring）**：创建/包装 Camera prim、应用 OmniSensorAPI，`.camera` 暴露光学参数。
- **CameraSensor（runtime）**：指定分辨率创建 Replicator render product，挂 annotator，`get_data()` 取数。

### 用法要点
```python
from isaacsim.sensors.experimental.rtx import RtxCamera, CameraSensor
camera = RtxCamera("/World/camera", tick_rate=30.0)      # tick_rate 限频
sensor = CameraSensor(camera, resolution=(480, 640), annotators=["rgb", "distance_to_image_plane"])
data, info = sensor.get_data("rgb")                       # 返回 (warp.array, info)
```
- **annotator**：`rgb` / `distance_to_camera`（射线距离）/ `distance_to_image_plane`（到像平面垂直距离）/ `semantic_segmentation` / `motion_vectors` / `depth_sensor_distance`。机械臂抓取更常用 `distance_to_camera`。深度两种语义单位都是米（stage 单位），实现时实测确认。
- **TiledCameraSensor**：多相机打包单个 tiled render product，对 RL/多环境效率高得多。
- **tick_rate**：`0`（默认）= 每仿真帧都渲染（最烧 GPU）；非 0 按频率独立渲染。VLA 不需要 60Hz 视觉 → 用 15~30Hz。
- **单目深度**：`SingleViewDepthCameraSensor`（单目视锥模拟立体深度），`set_enabled_post_processing(True)`；不要上立体/结构光模拟。
- 官方资产：`/Isaac/Sensors/RealSense/D455/rsd455.usd`（`RtxCamera.create(path, usd_path=...)`）。

### 8GB 显存纪律（本项目核心）
1. **分辨率即显存预算**：压到 VLA 所需最低（256×256 / 320×240）；从 640×480 → 256×256 像素降到 ~1/5。
2. **长宽比必须与 aperture 一致**（Omniverse 只支持方形像素）：改 render product 尺寸要连带改 horizontal/verticalAperture。
3. **三招省显存**：`TiledCameraSensor` 批处理 + `tick_rate` 限频 + 只挂 rgb/深度两个 annotator。
4. **抓帧回 CPU**：`get_data()` 默认返回 GPU 驻留 warp.array，训练循环里转 numpy/卸载，annotator 缓冲设 CPU 驻留（`camera_annotator_devices.py`），否则图像与渲染、VLA 前向抢显存。

## 三、机器人仿真控制与排障

### 关节控制
- `ArticulationController`（legacy）**已废弃** → `isaacsim.core.experimental.prims.Articulation`（Isaac Lab 3.0 的 `isaaclab.assets.Articulation` 同源上层）。
- 一个关节同一时刻**只能一种控制方式**（位置/速度/力矩三选一），不能混用。
- legacy `ArticulationAction` 中值为 0 的项会被当作"未驱动"——在 Isaac Lab 用 `joint_pos`/`joint_vel` 显式给更安全。
- 夹爪跟踪不准/指令漂移 → 先增大控制器 **stiffness 和 damping**。

### 物理 vs 渲染时钟（易出莫名慢放/快进）
- 物理步长由 Physics Scene 独立设置；**只改渲染帧率不会增加每帧物理子步数**。
- 三个时钟（loop / timeline timeCodesPerSecond / 物理 timeStepsPerSecond）要配套：`SimulationManager.setup_simulation(dt=...)` 与 `RenderingManager.set_dt(...)` 用同一 dt。
- Fixed Time Stepping（完整版默认）下重场景渲染跟不上 → 用物理步回调程序化驱动运动（保确定性），别依赖 keyframe 动画。

### 仿真"爆炸"/不稳定的排查顺序
1. **重叠碰撞体**（头号原因）→ 用 Collision Filters；动态碰撞只允许凸包/凸分解/box/sphere/SDF，triangle mesh 只用于静态。
2. **质量/惯量**：MassAPI 设真值（缺省按体积×1000kg/m³ 估算，常不合理）。
3. **关节阻尼/刚度**。
4. **时间步长过大**。

### 夹爪抓不住物体
- 提高手指与被抓物摩擦系数（通常 ≤1.0）。
- 检查物体/手臂质量（Mass Distribution Tool）。
- 增大夹爪控制器 stiffness/damping。

## 四、快速上手（GUI）
`isaac-sim.bat` 启动 → File>New → Create>Physics>Ground Plane + Lights>Distant Light + Shape>Cube → Play → 选中物体 Add "Rigid Body with Colliders Preset" 获得物理+碰撞。
排障速查：低帧率 UI 卡顿用 Ctrl+点击；`--/log/level=error` 减日志；`./isaac-sim.sh --reset-user` 清持久化帧率；Windows 关闭时线程未清理可忽略。

## 五、对本项目落地的关键结论
1. **项目工作走 Isaac Lab 高层框架**（`isaaclab.envs`/`assets`/`SimulationContext`），这些原生笔记用于下沉排障与自定义传感器。
2. **相机走新 RTX 实验 API**（`isaacsim.sensors.experimental.rtx`），不碰已废弃的 `isaacsim.sensors.camera`；8GB 下用 Tiled + tick_rate + 低分辨率 + CPU 缓冲四件套。
3. **Franka 驱动**：Isaac Lab 里选好 action mode 别中途换；关节数组与索引严格对应；物理不稳先查碰撞体/质量，抓不住先调摩擦/刚度。
4. **改 dt 要配套**：物理步长与渲染时钟用同一 dt，否则莫名慢放/快进。
5. **版本落差**：文档 API 版本高于/不同于本地，移植 snippet 前核对本地包名。
