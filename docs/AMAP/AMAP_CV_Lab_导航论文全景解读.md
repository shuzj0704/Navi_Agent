# AMAP CV Lab (Alibaba) 导航论文全景解读

> 本文档解读 AMAP CV Lab 在 Embodied Navigation 领域的 10 篇系列工作（2025.10 — 2026.03），梳理每篇的核心内容、技术演进脉络和相互关系。

---

## 总览

| # | 论文 | 时间 | 任务 | 核心创新 | VLM | 动作输出 |
|---|------|------|------|---------|-----|---------|
| 1 | CE-Nav | 2025.10 | 跨具身局部避障 | VelFlow (Normalizing Flow) + IL→RL | 无 | 体速度 (v_x, v_y, v_yaw) |
| 2 | Nav-R² | 2025.12 | ObjectNav | 双关系 CoT 推理 + SA-Mem | Qwen2.5-VL-7B | 离散动作 |
| 3 | AstraNav-Memory | 2025.12 | Lifelong ObjectNav | 20x 视觉压缩记忆 | Qwen2.5-VL-3B | 文本坐标 |
| 4 | OmniNav | 2026.01 | 统一4任务 | Fast-Slow 双系统 + 通用VL联训 | Qwen2.5-VL-3B | 5-step waypoints (FM) |
| 5 | FantasyVLN | 2026.01 | VLN 长程 | 隐式多模态 CoT | Qwen2.5-VL | 离散动作 |
| 6 | BridgeNav | 2026.02 | Out-to-In POI | Latent Intention + 光流感知 | Qwen2.5-VL-3B | 10-step waypoints |
| 7 | ABot-N0 | 2026.02 | 统一5任务 | Grand Unification + 最大数据引擎 | Qwen3-4B | 5-step waypoints (FM) |
| 8 | SocialNav | 2026.02 | 社会合规导航 | Brain-Action + SAFE-GRPO | Qwen2.5-VL-3B | 5-step waypoints (FM) |
| 9 | JanusVLN | 2026.02 | VLN-CE | 双隐式 KV-cache 记忆 | Qwen2.5-VL-7B | 离散动作 |
| 10 | NavForesee | 2026.03 | VLN-CE | 规划+世界模型预测统一 | Qwen2.5-VL-3B | 5-step waypoints (FM) |

FM = Flow Matching (DiT-based)

---

## 1. CE-Nav: Flow-Guided Reinforcement Refinement for Cross-Embodiment Local Navigation

- **作者**: Kai Yang*, Tianlin Zhang*, Zhengbo Wang* 等 | 通讯: **Zedong Chu**（AMAP Alibaba）
- **发表**: arXiv 2509.23203, 2025.10
- **项目**: AMAP CV Lab

### 做了什么

跨具身本体的局部避障导航框架。核心问题：如何让一个通用策略适配不同机器人（四足/双足/旋翼）而不需要从头训练？

### 输入/输出

- **输入**: 2D LiDAR scan（144 rays）+ 7D 机器人状态（目标方向+速度+距离）
- **输出**: 体速度指令 (v_x, v_y, v_yaw)

### 主要 Contributions

1. **IL-then-RL 范式**：解耦通用几何推理（VelFlow, 离线 IL）和具身特定动力学适配（Refiner, 在线 RL）
2. **VelFlow**：基于 Real-NVP 的条件归一化流，从 DWA 经典规划器的 10M 状态-动作对学习多模态速度分布，解决 "disastrous averaging" 问题
3. **6 小时适配新机器人**：冻结 VelFlow，仅训轻量 Refiner (PPO)，跨 5 种机器人（Go2/B2/H1/X20/飞行器）SOTA

### 关键结果

- Isaac Sim obstacle forest: SR 83%/SPL 0.78（N_o=500），显著超越纯 RL 和纯 IL 基线
- 仅 6h 训练（vs 纯 RL 52h 且 SR 更低）

### 在 AMAP 体系中的位置

**最底层——System 1 级别的局部控制器**。后续 SocialNav/ABot-N0 的 Action Expert 继承了 Flow-based 动作生成的思路（从 Normalizing Flow 升级为 Flow Matching DiT）。

---

## 2. Nav-R²: Dual-Relation Reasoning for Generalizable Open-Vocabulary Object-Goal Navigation

- **作者**: Wentao Xiang, Haokang Zhang 等 | 通讯: **Zedong Chu**（AMAP）/ **Yujiu Yang**（Tsinghua）
- **发表**: arXiv 2512.02400, 2025.12

### 做了什么

开放词汇 ObjectNav 的结构化 CoT 推理框架。显式建模两种关系：target-environment（目标在哪）和 environment-action（怎么走过去）。

### 输入/输出

- **输入**: 前视 RGB + 历史帧（SA-Mem 管理）+ 目标物体名称
- **输出**: 结构化 CoT 推理文本 + 离散导航动作

### 主要 Contributions

1. **双关系推理**：将 ObjectNav 决策分解为"目标在哪"（target-env）和"怎么走"（env-action）两步，用结构化 CoT 串联
2. **SA-Mem**（Similarity-Aware Memory）：基于帧间相似度的压缩和淘汰策略，无额外参数
3. **ObjectNav CoT 数据集**（~300K 样本）

### 关键结果

- HM3D-OVON Val-Unseen: SR 44.0%/SPL 18.0%（无深度图方法最佳），2Hz 推理

### 在 AMAP 体系中的位置

**ObjectNav 推理方法**。其 CoT 推理思路影响了 ABot-N0 的 Object-Goal Reasoning 数据构建。

---

## 3. AstraNav-Memory: Contexts Compression for Long Memory

- **作者**: Botao Ren, Junjun Hu（Project Lead）, Xinda Xue 等 | 通讯: **Junjun Hu**（AMAP）
- **发表**: arXiv 2512.21627, 2025.12

### 做了什么

解决 Lifelong Navigation 的长程视觉记忆问题。用高效压缩模块（DINOv3 + PixelUnshuffle + Conv）将每帧图像压缩到 30 tokens（20x 压缩），直接注入 VLM 上下文，在单个 context 中容纳数百帧历史。

### 输入/输出

- **输入**: RGB 四视角图像序列（最多 300 帧 × 30 tokens = 9000 tokens）+ 相机位姿 + 指令
- **输出**: 文本坐标（frontier 或目标位置）

### 主要 Contributions

1. **Plug-and-play ViT-native 视觉 tokenizer**：DINOv3 + PixelUnshuffle + Conv，20x 压缩，保留空间语义
2. **Image-centric 长程隐式记忆框架**：端到端耦合导航策略，无需外部图结构
3. GOAT-Bench SR +15.5%，HM3D-OVON SR +21.7%

### 关键结果

- GOAT-Bench Val-Unseen: SR 62.7%/SPL 56.9%
- HM3D-OVON Val-Unseen: SR 62.5%/SPL 34.8%

### 在 AMAP 体系中的位置

**记忆压缩方案**。其 20x 视觉压缩思路为 OmniNav 的 image memory 和 ABot-N0 的 episodic visual memory 提供了基础。

---

## 4. OmniNav: A Unified Framework for Prospective Exploration and Visual-Language Navigation

- **作者**: Xinda Xue*, Junjun Hu*†（共一，Project Lead）等 | 通讯: **Junjun Hu / Jintao Chen**（AMAP / PKU）
- **发表**: arXiv 2509.25687v3, 2026.01
- **项目**: https://astra-amap.github.io/omninav.github.io/

### 做了什么

统一导航框架——一个模型同时处理 instruct-goal、point-goal、object-goal 和 frontier exploration 四种任务。Fast-Slow 双系统 + Flow Matching waypoint policy。

### 输入/输出

- **Fast System**: 历史 RGB 帧（20 帧采样）+ 任务文本/坐标 → 5-step continuous waypoints
- **Slow System**: 长历史 + frontier 图像 + occupancy map → CoT 推理 → subgoal 坐标

### 主要 Contributions

1. **统一多任务框架**（4 种目标模态 + frontier 探索），单一策略和训练
2. **Fast-Slow 双系统 + Central KV-cache Memory**
3. **关键发现：泛化瓶颈不在导航策略，而在通用 VL 理解**——混入 5M 通用数据后 SR 显著提升

### 关键结果

- R2R-CE: SR 69.5%（+4.4%），RxR-CE: SR 73.6%（+4.3%）
- HM3D-OVON（with slow）: SR 59.2%（+18.4%）
- CityWalker: MAOE 7.8

### 在 AMAP 体系中的位置

**多任务统一的中枢框架**。ABot-N0 直接继承其多任务统一思路并扩展到 5 类任务。

---

## 5. FantasyVLN: Unified Multimodal Chain-of-Thought Reasoning for Vision-Language Navigation

- **作者**: Jing Zuo, Lingzhou Mu, Fan Jiang（Project Lead）等 | 通讯: **Fan Jiang**（AMAP Fantasy AIGC）/ **Yonggang Qi**（BUPT）
- **发表**: arXiv 2601.13976, 2026.01

### 做了什么

首个统一 Text CoT + Visual CoT + Multimodal CoT 的隐式推理框架。训练时学多种 CoT 模式并对齐，推理时直接用 non-CoT 模式（无延迟开销）。用 VAR 将"想象的未来观测"压缩到 30 tokens。

### 输入/输出

- **输入**: 多视角 RGB + 自然语言指令
- **输出**: 训练时：CoT 文本/压缩视觉 CoT + 动作；推理时：直接动作（无 CoT 文本输出）

### 主要 Contributions

1. **首个统一 text/visual/multimodal CoT 的隐式推理框架**（train-with-CoT, infer-without-CoT）
2. **CompV-CoT**：用 VAR 将想象观测压缩到 30 tokens
3. **Cross-mode alignment**：将所有 CoT 模式对齐到 non-CoT hidden state

### 关键结果

- LH-VLN: SR 2.44（vs Aux-Think 0.65），推理速度 1.03 actions/sec（vs CoT-VLA 0.19）

### 在 AMAP 体系中的位置

**推理效率路线**（Fantasy AIGC 子团队）。隐式 CoT 思路与 NaviAgent 的 Latent Adaptive Reasoning 异曲同工。

---

## 6. BridgeNav: Bridging the Indoor-Outdoor Gap — Vision-Centric Instruction-Guided Embodied Navigation

- **作者**: Yuxiang Zhao*, Yirong Yang*, Yanqing Zhu（Project Leader）等 | 通讯: **Mu Xu**（AMAP）
- **发表**: arXiv 2602.06427, 2026.02

### 做了什么

定义 out-to-in prior-free 导航新任务——仅凭视觉和简短指令（"go to Starbucks"），从室外导航到目标建筑入口，无 GPS/地图。

### 输入/输出

- **输入**: RGB 前视图序列 + 文本指令（POI 名称）
- **输出**: 10-step waypoint 序列 (x, y, z, yaw)

### 主要 Contributions

1. **定义 out-to-in prior-free 导航新任务**
2. **Latent Intention Inference**：根据距离动态关注不同视觉区域（远看招牌→中看标识→近看入口）
3. **首个 out-to-in 数据集**（55K 轨迹，10M+ 帧，trajectory-conditioned video generation 生成）

### 关键结果

- SR(0.1m) 33.82%（+80.1% vs OmniNav），SR(0.3m) 89.55%

### 在 AMAP 体系中的位置

**定义了 POI-Goal 导航任务**。ABot-N0 直接将 POI-Goal 作为第 5 类统一任务，在 BridgeNav 数据集上评测。

---

## 7. ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation

- **作者**: AMAP CV Lab 团队 | 通讯: **Zedong Chu / Shichao Xie**（AMAP）
- **发表**: arXiv 2602.11598, 2026.02
- **项目**: https://amap-cvlab.github.io/ABot-Navigation/ABot-N0/

### 做了什么

**集大成之作**——首个覆盖 5 类导航任务的统一 VLA 基础模型：Point-Goal、Object-Goal、Instruction-Following、POI-Goal、Person-Following。构建了迄今最大规模导航数据引擎。

### 输入/输出

- **输入**: RGB 全景/前视图 + 视觉历史 + 目标（文本/坐标/POI 名/人物描述）+ 推理 prompt
- **输出**: CoT 推理文本 + 可通行区域 polygon + VQA 回答 + 5-step continuous waypoints

### 主要 Contributions

1. **Grand Unification**：首个 5 任务统一 VLA，7 个 benchmark SOTA
2. **最大导航数据引擎**：7,802 个 3D 场景（10.7 km²），16.9M 专家轨迹 + 5M 认知推理样本
3. **可部署 Agentic Navigation System**：Agentic Planner (CoT) + Map-as-Memory (层级拓扑) + Neural Controller (ABot-N0)，在 Unitree Go2 + Jetson Orin NX 上 2Hz VLA + 10Hz 闭环

### 架构

```
Universal Multi-Modal Encoder
  ├── Flexible Vision Interface（全景/前视 RGB + 视觉历史）
  ├── Navigation Target Encoder（坐标/文本/POI/人物目标）
  └── Reasoning Task Encoder（CoT/VQA/可通行区域 prompt）
            ↓
   Cognitive Brain (Qwen3-4B LLM)
  ├── Reasoning Head → CoT / VQA / 可通行区域文本
  └── Action Head → VLM features
            ↓
   Action Expert (Flow Matching DiT)
  └── 5-step continuous waypoints (x, y, θ)
```

### 训练（3 阶段）

| 阶段 | 数据 | 训练内容 |
|------|------|---------|
| Phase 1: Cognitive Warm-up | 5M 推理数据 | 仅训 Cognitive Brain，冻结 Action Expert |
| Phase 2: Unified Sensorimotor SFT | 16.9M 轨迹 + 推理混合 | Brain + Action Expert 联合训练 |
| Phase 3: SAFE-GRPO | SocCity 环境 | Flow-based RL 社会合规对齐 |

### 关键结果

| Benchmark | 指标 | ABot-N0 | 此前 SOTA |
|-----------|------|:---:|:---:|
| CityWalker (开环) | MAOE | **7.8** | 15.2 |
| SocNav (闭环) | SR | **88.3%** | 65.0% |
| R2R-CE | SR | **66.4%** | 65.1% |
| HM3D-OVON | SR | **54.0%** | 47.0% |
| BridgeNav | SR(0.1m) | **32.14%** | 18.78% |
| EVT-Bench | SR | **86.9%** | - |

### 在 AMAP 体系中的位置

**所有前作的集大成者**。整合了 SocialNav 的 Brain-Action 架构 + SAFE-GRPO、OmniNav 的多任务统一、BridgeNav 的 POI-Goal 任务、Nav-R² 的 ObjectNav CoT、AstraNav-Memory 的视觉记忆思路。

---

## 8. SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation

- **作者**: Ziyi Chen*, Yingnan Guo*（共一）等 | 通讯: **Zedong Chu†**（AMAP）
- **发表**: arXiv 2511.21135, 2026.02
- **项目**: https://amap-eai.github.io/SocialNav/

### 做了什么

首个社会规范感知的导航基础模型——不仅走到目标，还要遵守社会规范（走人行道、不横穿草坪）。

### 输入/输出

- **输入**: RGB 前视图序列 + 2D 位置历史 + 目标坐标
- **输出**: CoT 推理 + 可通行区域 polygon + 5-step waypoints (Flow Matching)

### 主要 Contributions

1. **Brain-Action 分层架构**：VLM Brain（社会规范理解）+ Action Expert DiT（合规轨迹生成）
2. **SAFE-GRPO**：首个 flow-based RL 框架（SDE 替代 ODE），norm-aware reward 强制社会合规
3. **SocNav Dataset (7M)** + **SocNav Benchmark**（9 个 3DGS 场景 + Isaac Sim 物理仿真）

### 关键结果

- SocNav Bench: SR 86.1%, DCR 82.5%（+38% SR, +46% DCR vs CityWalker）
- 真实世界 3 场景: SR 85.0%

### 在 AMAP 体系中的位置

**Brain-Action 架构的首创者 + SAFE-GRPO 的发源地**。ABot-N0 直接继承此架构和 RL 方案。

---

## 9. JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for VLN

- **作者**: Shuang Zeng, Dekang Qi 等 | 通讯: **Xing Wei**（XJTU）/ **Ning Guo**（AMAP）
- **发表**: ICLR 2026, arXiv 2509.22548

### 做了什么

提出双隐式神经记忆：将 VLN 记忆从显式序列转为固定大小 KV-cache，分离语义记忆和空间几何记忆。仅用 RGB（无深度）在 VLN-CE 上 SOTA。

### 输入/输出

- **输入**: RGB 前视视频流 + 自然语言指令
- **输出**: 离散动作 {Forward, Turn Left, Turn Right, Stop}

### 主要 Contributions

1. **双隐式记忆范式**：spatial-geometric（VGGT）+ visual-semantic（Qwen2.5-VL），固定大小 KV-cache，增量更新
2. **首次将 3D spatial geometry foundation model (VGGT) 引入 VLN**
3. VLN-CE SOTA，仅用 RGB

### 关键结果

- R2R-CE Val-Unseen: SR 65.2%/SPL 56.8%（+10.5-35.5% SR vs 同类）

### 在 AMAP 体系中的位置

**记忆方案的独立路线**（KV-cache 隐式记忆），与 AstraNav-Memory（压缩 token 记忆）是互补方案。

---

## 10. NavForesee: A Unified Vision-Language World Model for Hierarchical Planning and Dual-Horizon Navigation Prediction

- **作者**: Fei Liu*, Shichao Xie*（共一）等 | 通讯: **Zedong Chu / Xiaolong Wu**（AMAP）
- **发表**: arXiv 2512.01550v2, 2026.03

### 做了什么

将 VLM 规划和世界模型预测统一到单一模型——不仅做"该往哪走"的决策，还做"走过去会看到什么"的预测。

### 输入/输出

- **输入**: 全景 RGB + 指令 + 位姿编码 + dream queries
- **输出**: 分层语言规划（summary + plan + actions）+ 双时间尺度环境特征预测（短期 depth/semantic + 长期 milestone features）+ 5-step waypoints

### 主要 Contributions

1. **统一 VLM 规划与世界模型预测**的单一 VLN 框架
2. **分层语言规划**：milestone-based sub-instruction decomposition + 进度追踪
3. **双时间尺度预测**：短期预测（k 步内 depth + DINOv2/SAM semantic）+ 长期预测（到下一 milestone）

### 关键结果

- R2R-CE: SR 66.2%（+1.1%），OSR 78.4%（+10.9%）

### 在 AMAP 体系中的位置

**预测式导航的独立路线**。将"世界模型"概念引入 VLN，与主线的 Brain-Action 架构互补。

---

## 技术演进脉络

### 时间线

```
2025.10  CE-Nav          ← 底层跨具身避障（VelFlow）
2025.12  Nav-R²          ← ObjectNav CoT 推理
2025.12  AstraNav-Memory ← 长程视觉记忆压缩
2026.01  OmniNav         ← 统一4任务 + Fast-Slow 双系统
2026.01  FantasyVLN      ← 隐式多模态 CoT
2026.02  BridgeNav       ← 定义 POI-Goal 任务
2026.02  SocialNav       ← Brain-Action 架构 + SAFE-GRPO
2026.02  ABot-N0         ← ★ 集大成：5任务统一 VLA 基础模型
2026.02  JanusVLN        ← 双隐式KV-cache记忆（ICLR'26）
2026.03  NavForesee      ← 规划+世界模型预测统一
```

### 技术传承关系

```
CE-Nav (底层 VelFlow)
  └──→ SocialNav (升级为 Flow Matching DiT + SAFE-GRPO)
        └──→ ABot-N0 (直接复用 Brain-Action + SAFE-GRPO)

Nav-R² (ObjectNav CoT)
  └──→ ABot-N0 (Object-Goal Reasoning 数据)

AstraNav-Memory (视觉压缩记忆)
  └──→ OmniNav (image memory bank)
        └──→ ABot-N0 (episodic visual memory)

OmniNav (统一4任务 + Fast-Slow)
  └──→ ABot-N0 (扩展到5任务，继承 Fast-Slow 架构)

BridgeNav (定义 POI-Goal 任务 + 数据集)
  └──→ ABot-N0 (POI-Goal 成为第5类任务)

JanusVLN (双隐式记忆)     ← 独立路线
FantasyVLN (隐式 CoT)    ← 独立路线 (Fantasy AIGC Team)
NavForesee (世界模型预测)  ← 独立路线
```

### 共性设计模式

| 模式 | 具体表现 | 使用的论文 |
|------|---------|----------|
| **Brain-Action 架构** | VLM 做语义推理 + Action Expert (DiT/Flow) 做轨迹生成 | SocialNav → OmniNav → ABot-N0 → NavForesee |
| **Flow Matching 动作头** | DiT-based conditional flow matching 生成连续 waypoints | CE-Nav(NF) → SocialNav → OmniNav → ABot-N0 → NavForesee |
| **Qwen2.5-VL 骨干** | 3B 或 7B 作为 VLM backbone | 除 CE-Nav 和 ABot-N0(Qwen3-4B) 外全部 |
| **3 阶段训练** | Pre-train → SFT → RL (SAFE-GRPO) | SocialNav → ABot-N0 |
| **CoT 推理** | 显式 text CoT 或 隐式 latent CoT | Nav-R², SocialNav, OmniNav(slow), FantasyVLN, ABot-N0, NavForesee |
| **可通行区域预测** | VLM 输出 polygon 标注安全通行区域 | SocialNav → ABot-N0 |

---

## 对 NaviAgent 的综合启示

### NaviAgent 与 AMAP 体系的定位对比

| 维度 | ABot-N0 (AMAP 最强) | NaviAgent (ours) |
|------|:---:|:---:|
| 任务 | 5 任务统一（室内+室外） | 跨环境长程导航（室内→室外→室内） |
| 架构 | Brain-Action (VLM + DiT FM) | System 2 (VLM + NaVocab + Memory) + System 1 (DiT) |
| 工具调用 | 无（纯 VLM 推理） | **Learned tool tokens**（可微） |
| 记忆 | Map-as-Memory（外部拓扑图） | **Differentiable Memory Bank**（VLM context tokens） |
| 推理 | 显式 CoT + SAFE-GRPO | **Latent thinking tokens**（渐进压缩） |
| 跨环境过渡 | 通过 POI-Goal 间接处理 | **专门设计的 GPS 过渡 + 模式切换** |
| 动作输出 | Flow Matching waypoints | **Pixel goal + Latent goal → DiT** |
| VLM | Qwen3-4B | Qwen3.5-9B |
| Depth 利用 | 无 | **DPE (Depth Positional Encoding)** |

### NaviAgent 的差异化核心

1. **Tool tokens 范式**：ABot-N0 的所有"工具"是 VLM 文本推理输出（CoT → 行动），NaviAgent 用 learned token embedding 做可微工具调用
2. **Differentiable Memory Bank**：ABot-N0 用外部 Map-as-Memory（拓扑图），NaviAgent 用 VLM 上下文中的 learned embedding tokens（端到端可微）
3. **跨环境过渡机制**：ABot-N0 将室内外视为不同任务类型切换，NaviAgent 通过 NaVocab (GPS 状态 token) + Memory + `<tool:route>` 实现连续过渡
4. **Latent Reasoning**：ABot-N0 用显式 text CoT（推理时有文本输出），NaviAgent 用 latent tokens（推理时无文本，避免 reasoning collapse）

### 可借鉴的设计

| 来自 | 借鉴内容 |
|------|---------|
| **SocialNav/ABot-N0** | SAFE-GRPO (flow-based RL) → NaviAgent Stage 3 RFT 的直接参考 |
| **OmniNav** | 通用 VL 数据联训提升泛化 → NaviAgent SFT 时混入通用数据 |
| **AstraNav-Memory** | 20x 视觉压缩 → NaviAgent Memory Bank 的 token 效率参考 |
| **FantasyVLN** | 隐式 CoT (train-with / infer-without) → NaviAgent Latent Adaptive Reasoning |
| **BridgeNav** | POI-Goal 任务定义 + Latent Intention → NaviAgent 的跨环境 POI 导航场景 |
| **ABot-N0 Data Engine** | 7,802 场景 / 16.9M 轨迹的数据规模 → NaviAgent 数据流水线的目标参考 |
