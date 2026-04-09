# 室外城市仿真器论文解读

> 两篇来自 UCLA Bolei Zhou 团队的城市仿真平台工作，一篇聚焦 real-to-sim 场景生成（UrbanVerse），一篇聚焦程序化场景 + benchmark 任务定义（Urban-Sim）。两者共享作者和代码生态（IsaacSim + IsaacLab），互为补充。

---

## 总览对比

| 维度 | UrbanVerse (ICLR'26) | Urban-Sim (CVPR'25 Highlight) |
|------|:---:|:---:|
| **核心贡献** | Real-to-sim 场景生成系统 | 程序化仿真平台 + Benchmark 任务定义 |
| **场景生成方式** | 从 YouTube 视频自动重建（data-driven） | 程序化生成（hierarchical procedural） |
| **场景引擎** | IsaacSim 4.5.0 | IsaacSim (Omniverse) + PhysX 5 |
| **场景数量** | 160 训练 + 12 评测（CraftBench） | **无限**（程序化，可生成任意数量） |
| **资产库** | UrbanVerse-100K (102K 物体) | 15,000+ 物体 |
| **场景真实度** | **高**（grounded in real-world video） | 中等（程序化模板） |
| **动态 Agent** | ORCA 行人/骑行者（GPU 加速） | ORCA 行人/骑行者 + 车辆（GPU 加速） |
| **训练性能** | 未公开 FPS | **1,800+ FPS**（单 L40S GPU） |
| **支持机器人** | COCO 轮式 + Go2 四足 | COCO + Go2 + B2-W 轮足 + G1 人形（4 种） |
| **Benchmark 任务** | 导航评测（AutoBench + CraftBench） | **8 个任务**：4 Locomotion + 3 Navigation + 1 Traverse |
| **训练方法** | PPO (rl-games) | PPO (rl-games) |
| **开源状态** | 资产库+场景已开源，训练代码未开源 | **代码已开源** (github.com/metadriverse/urban-sim) |
| **论文** | arXiv 2510.15018 | arXiv 2505.00690 |

---

## 1. UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos

> ICLR 2026 | arXiv 2510.15018
> https://github.com/VAIL-UCLA/UrbanVerse

- **作者**: Mingxuan Liu*, Honglin He*（共一）等 | 通讯: **Bolei Zhou**（UCLA）
- **项目**: https://urbanverseproject.github.io/

### 做了什么

UrbanVerse 是一个 **data-driven real-to-sim 系统**——从 YouTube 城市旅游视频自动生成物理仿真场景。不是一个导航模型，而是一个**导航训练平台**。

### 主要 Contributions

1. **UrbanVerse-100K 资产库**：102,444 个 GLB 3D 物体（659 类别，33 种语义/物理/功能属性标注）+ 288 种路面材质 + 306 张 HDRI 天空图。三级城市本体论（OpenStreetMap 扩展）
2. **UrbanVerse-Gen 场景生成管线**：从 uncalibrated RGB 视频自动生成 IsaacSim 交互式场景。三阶段：蒸馏（MASt3R SfM + YoloWorld + SAM2 → 3D Scene Graph）→ 物化（CLIP + DINOv2 检索 digital cousin）→ 生成（IsaacSim 实例化）
3. **Data Scaling Power Law**：证明场景数翻倍 → 导航 SR 持续提升，为"用更多场景训练"提供了理论支撑

### 输入/输出

- **系统输入**: YouTube 城市步行视频（uncalibrated RGB）
- **系统输出**: IsaacSim 中的交互式仿真场景
- **导航策略输入**: RGB 图像 + 相对目标位置（无地图）
- **导航策略输出**: 离散动作（PPO 训练）

### 关键结果

- 场景质量：93.1% 物体类别正确恢复，位置偏差仅 1.4m
- 导航策略：PPO 在 UrbanVerse 训练 → 零样本 sim2real **SR 89.7%**
- 真机：337m 长程任务仅 2 次人工干预

### 开源资源

| 资源 | 大小 | 状态 |
|------|:---:|:---:|
| UrbanVerse-100K 资产库 | 1.18TB | 可下载 |
| CraftBench 12 评测场景 | 16.8GB | 可下载（需申请）|
| 160 训练场景 | - | **未公开** |
| Gen pipeline 代码 | - | **未公开** |
| 导航训练代码 | - | **未公开** |

---

## 2. Urban-Sim: Towards Autonomous Micromobility through Scalable Urban Simulation

> CVPR 2025 Highlight | arXiv 2505.00690
> https://metadriverse.github.io/urban-sim/
> https://github.com/metadriverse/urban-sim

- **作者**: Wayne Wu*, Honglin He*（共一）, Chaoyuan Zhang, Jack He, Seth Z. Zhao, Ran Gong, Quanyi Li, **Bolei Zhou** | 通讯: **Bolei Zhou**（UCLA）
- **项目**: https://metadriverse.github.io/urban-sim/

### 做了什么

Urban-Sim 是一个**高性能程序化城市仿真平台** + **8 任务 benchmark**，专为自主微出行（配送机器人、电动轮椅、四足等）设计。与 UrbanVerse 的 real-to-sim 互补——Urban-Sim 用程序化生成实现**无限场景**和**极高训练吞吐量**。

### 主要 Contributions

1. **URBAN-SIM 仿真平台**：基于 IsaacSim + PhysX 5，三大关键设计：
   - **Hierarchical Urban Generation**：4 级程序化生成（街区连接 → 地面规划 → 地形生成（Wave Function Collapse）→ 物体放置），可生成无限多样的城市场景
   - **Interactive Dynamics Generation**：GPU 加速的 ORCA 多 agent 路径规划（行人、骑行者实时交互），比 CPU ORCA 无 CPU-GPU 数据传输瓶颈
   - **Asynchronous Scene Sampling**：每个 GPU 环境加载不同场景（异步采样），比同步采样（同一场景）提升 26.3%+，单 L40S GPU 达 **1,800+ FPS**

2. **URBAN-BENCH**：8 个任务覆盖微出行的三大核心技能：
   - **Urban Locomotion（4 任务）**：LocoFlat（平地）、LocoSlope（斜坡）、LocoStair（楼梯）、LocoRough（粗糙地面）
   - **Urban Navigation（3 任务）**：NavClear（开阔区通行）、NavStatic（静态障碍物避障）、NavDynamic（动态障碍物避障）
   - **Urban Traverse（1 任务）**：公里级长程导航（A→B，>1km），支持 Human-AI 共享自主方案

3. **跨机器人形态评测**：4 种机器人（COCO 轮式、Go2 四足、B2-W 轮足、G1 人形）在 8 个任务上的完整 benchmark，揭示了不同机械结构的独特行为模式

### 输入/输出

- **仿真输入**: 程序化生成参数 → 自动构建城市场景
- **导航策略输入**: RGBD 图像 + 相对目标位置
- **导航策略输出**: 连续速度控制（PPO 训练）
- **训练接口**: IsaacLab `ManagerBasedRLEnv`，兼容 rl-games / Stable Baselines3

### 核心架构细节

**场景生成 4 阶段**：
1. Block Connection：采样街区类型（直道、弯道、环岛、交叉口、T 字路口等）并连接
2. Ground Planning：划分功能区（人行道、斑马线、广场、建筑、植被），随机化参数
3. Terrain Generation：Wave Function Collapse 生成地形（平地、台阶、斜坡、粗糙路面）
4. Object Placement：根据功能区放置静态物体（树木、长椅、垃圾桶、路灯等），15,000+ 资产

**训练性能**：
- 单 L40S GPU：256 并行环境，1,800-2,600 FPS（含 RGBD 渲染）
- 异步场景采样 vs 同步：场景数从 1→256 时性能差距显著拉开
- 训练场景数 1→1024：SR 从 5.1% 提升至 83.2%

**4 种机器人**：

| 机器人 | 类型 | 特点 |
|--------|------|------|
| COCO | 轮式配送机器人 | 开阔区高效，楼梯无法通过 |
| Unitree Go2 | 四足 | 运动最平滑，楼梯/斜坡强 |
| Unitree B2-W | 轮足混合 | 最全面，距离+时间+平衡全面 |
| Unitree G1 | 人形 | 平地/斜坡最稳，狭窄空间灵活 |

### 关键结果

**Urban Navigation Benchmark**:

| 任务 | 最佳机器人 | SR |
|------|---------|:---:|
| NavClear | 轮式 COCO | 97.6% |
| NavStatic | 人形 G1 | 77.9% |
| NavDynamic | 人形 G1 | 79.2% |

**Urban Traverse（公里级）**:
- 纯 AI 模式：最低人力成本但最多碰撞
- Human-AI Mode 1（调度导航+运动模型）：平衡性最好
- 人类模式：最安全但成本最高

**Scaling 实验**:
- 异步采样 + 场景数 1024 → **SR 83.2%**（vs 1 场景 5.1%）

### 开源状态

| 资源 | 状态 |
|------|:---:|
| Urban-Sim 平台代码 | **已开源** (github.com/metadriverse/urban-sim) |
| URBAN-BENCH 8 任务定义 | **已开源** |
| 4 种机器人配置 | **已开源** |
| 训练脚本 (PPO) | **已开源** |
| 场景生成 pipeline | **已开源** |
| 3D 资产库 (15K) | **已开源** |

**这是目前唯一完整开源了训练代码的城市导航仿真平台。**

---

## 两者的关系与互补

```
Urban-Sim（程序化）                    UrbanVerse（real-to-sim）
┌─────────────────────┐              ┌─────────────────────┐
│ 无限场景生成          │              │ 真实世界街景重建      │
│ 1,800+ FPS 高吞吐    │              │ 高保真度（视频驱动）   │
│ 8 任务 benchmark     │              │ 12 评测场景          │
│ 4 种机器人            │              │ 102K 资产库          │
│ 代码完全开源          │              │ 资产开源/代码未开源    │
└────────┬────────────┘              └────────┬────────────┘
         │                                    │
         └──────── 共享生态：IsaacSim + IsaacLab + PPO ────────┘
                           │
                   ┌───────┴───────┐
                   │  NaviAgent    │
                   │  室外训练用    │
                   │  Urban-Sim    │
                   │  室外评测用    │
                   │  CraftBench   │
                   └───────────────┘
```

**对 NaviAgent 的策略建议**：

| 阶段 | 用什么 | 原因 |
|------|--------|------|
| **开发阶段** | **Urban-Sim**（程序化场景） | 代码完全开源，可直接添加机器人跑导航，1,800 FPS 训练快 |
| **评测阶段** | **CraftBench**（UrbanVerse 12 场景） | 真实世界 grounded 场景，比程序化更能体现 sim2real 能力 |
| **数据扩展** | 两者混合 | Urban-Sim 提供量（无限场景），UrbanVerse 提供质（真实分布） |
| **真机部署** | UrbanVerse 160 场景（等开源） | Real-to-sim 场景 sim2real gap 最小 |

---

## 与 NaviAgent 的关联

- **Urban-Sim 是 NaviAgent 室外开发的首选平台**：代码完全开源，支持 4 种机器人，IsaacLab 标准接口，可直接集成 NaviAgent 的 System 1 DiT
- **CraftBench 作为高质量评测环境**：12 个场景已下载到 `/home/shu22/navigation/urban_verse/CraftBench/`
- **URBAN-BENCH 的 8 任务设计**为 NaviAgent 的室外评估提供了现成的任务定义（NavClear/NavStatic/NavDynamic + Traverse）
- 两者都用 **PPO + rl-games**，与 NaviAgent 的 Stage 3 RFT 兼容
- 两者都不涉及 **VLM 推理**——NaviAgent 的 System 2 token 架构 + Agent 推理是差异化核心
