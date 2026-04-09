# OmniNav: A Unified Framework for Prospective Exploration and Visual-Language Navigation

> arXiv 2509.25687, 2026.01 | Xinda Xue*, Junjun Hu*† 等 | 通讯: Junjun Hu / Jintao Chen (Amap Alibaba / PKU)

- **作者**: Xinda Xue*, Junjun Hu*†（共一，Hu 为 Project Lead）等 | 通讯: **Junjun Hu / Jintao Chen**（Amap Alibaba / PKU）
- **发表**: arXiv 2509.25687, 2026.01
- **项目**: https://astra-amap.github.io/omninav.github.io/

### 做了什么

OmniNav 是一个**统一导航框架**——用**一个模型、一套训练**同时处理 instruct-goal（VLN）、point-goal、object-goal 和 frontier-based exploration 四种任务。核心创新是 fast-slow 双系统 + flow-matching waypoint policy + 大规模通用 VL 数据联训。

### 主要 Contributions

1. **统一架构**：一个训练框架和策略，同时支持 point-goal、object-goal、instruct-goal 和 frontier-based exploration 四种目标模态。现有方法通常只针对单一任务定制，跨任务迁移差。OmniNav 用统一的坐标 token + 图像 token 输入格式 + flow-matching waypoint 输出格式，实现四任务共享训练。

2. **Fast-Slow 端到端协作 + Central Memory**：Fast System（5Hz）做短时视觉感知 → 连续 waypoint 生成（flow matching DiT）；Slow System（低频）做长时 frontier 推理 → CoT 选择 subgoal 坐标。两者通过 KV cache 形式的 Central Memory 共享时空上下文，既保证局部敏捷又保证全局一致。**这是导航领域首个将 fast-slow 协作、central memory 和 flow-matching policy 统一在 VLM 上的端到端系统。**

3. **通用 VL 数据联训提升泛化**：发现**导航性能的瓶颈不是策略学习本身，而是对通用指令和开放词汇物体的理解能力**。将 5M 通用 VL 数据（caption/QA/OCR/grounding）与 4M 导航数据联合训练后，SR 显著提升。这挑战了"导航只需导航数据"的假设。

### 核心发现（Insight）

> "The primary bottleneck lies not in navigation policy learning per se, but in robust understanding of general instructions and open-vocabulary objects."

导航策略本身容易学，**泛化的真正瓶颈是 VLM 对通用语言和视觉的理解能力**——加入 image captioning、visual grounding、OCR 等非导航数据后，导航成功率大幅提升。这对 NaviAgent 有重要启示：微调时不能只用导航数据，需要混入通用 VL 数据防止 catastrophic forgetting。

### 输入/输出

- **Fast Thinking System 输入**: 历史 RGB 帧（最近 20 帧采样）+ 任务指令/坐标 → VLM 处理
- **Fast Thinking System 输出**: 5-step continuous waypoints $w_t^{(i)} = (x^{(i)}, y^{(i)}, \sin\theta^{(i)}, \cos\theta^{(i)}, c^{(i)})$，其中 $c$ 是 arrive flag
- **Slow Thinking System 输入**: 长历史观测 + frontier 图像 + 3D occupancy map
- **Slow Thinking System 输出**: CoT 推理 → 选择 frontier/subgoal 的坐标 → 传给 Fast System 执行

### 核心架构

**Fast Thinking System（5Hz）**：

- VLM backbone: Qwen2.5-VL-3B-Instruct
- 坐标 token: sub-goal/frontier 坐标通过 MLP → dense embeddings
- 图像 token: 历史帧经 ViT 编码，时空采样
- 动作 head: **Conditional Flow Matching (DiT)**——不是自回归 text token，而是 diffusion 生成 5 步连续 waypoints
- KV cache: 流式推理，避免重复计算

**Slow Thinking System（低频调用）**：

- 用同一个 VLM（共享参数）
- 输入: 当前帧 + frontier 图像 + 占据图坐标
- 输出: CoT 推理文本 → 选择最优 frontier/subgoal 的坐标
- 仅在需要高层规划时调用（ObjectNav 探索、长程 VLN 路口决策）

**Central Memory**: KV cache 形式的共享记忆，连接 fast 和 slow 系统

### 训练数据（13M 样本）

| 类型                   | 规模 | 来源                                     |
| ---------------------- | :---: | ---------------------------------------- |
| Object-goal            | 1.5M | HM3D OVON（shortest path）               |
| Point-goal             |  1M  | CityWalker YouTube 视频（DPVO 恢复姿态） |
| Instruct-goal          | 0.75M | R2R-CE, RxR-CE（Habitat）                |
| Exploration (frontier) | 0.7M | HM3D frontier exploration                |
| Embodied QA            | 0.2M | ScanQA, R2R-EnvDrop                      |
| General MLLM           |  5M  | Caption/QA/OCR/Chart/Coding/Math         |
| Grounding & Referring  |  3M  | RefCOCO, Objects365                      |

**关键发现**: 导航本身学得快，**泛化的瓶颈在通用 VL 理解能力**（加 5M 通用数据后 SR 显著提升）

### 训练流程

- Stage 1: 自回归训练（离散 action chunks + 通用 VL 数据），96 H20 GPU, 120h
- Stage 2: 附加 flow-matching head 做连续 waypoint 训练，64 H20 GPU, 48h

### 关键结果

| Benchmark         | 指标 |     OmniNav     |     此前 SOTA     |
| ----------------- | ---- | :-------------: | :----------------: |
| R2R-CE Val-Unseen | SR   | **69.5%** | 65.1% (CorrectNav) |
| RxR-CE Val-Unseen | SR   | **73.6%** | 69.3% (CorrectNav) |
| HM3D-OVON         | SR   | **66.0%** |   47.0% (+18.4%)   |
| CityWalker        | MAOE |  **7.8**  | 11.5 (CityWalker) |

真机部署 5Hz，完成室内+室外导航任务。

### OmniNav vs InternVLA-N1 (DualVLN) 双系统架构对比

两者都是"慢规划 + 快执行"的双系统，但设计哲学有本质差异：

| 维度 | InternVLA-N1 / DualVLN | OmniNav |
|------|:---:|:---:|
| **System 2（慢系统）的角色** | VLM 生成 pixel goal + latent goal | VLM 生成 CoT 推理 → 选择 frontier/subgoal 坐标 |
| **System 1（快系统）的角色** | DiT 独立模型，接收 goal → 生成轨迹 | **同一个 VLM** 的 flow-matching head，接收 subgoal → 生成 waypoints |
| **是否共享参数** | **不共享**（System 2 是 VLM，System 1 是独立 DiT） | **共享 VLM backbone**（fast 和 slow 用同一个 Qwen2.5-VL-3B） |
| **慢系统输出格式** | pixel goal (u,v) + latent embedding | **坐标文本**（frontier/subgoal 的 (x,y) 坐标）+ CoT 推理文本 |
| **快系统输出格式** | 连续轨迹（DiT denoising） | 5-step waypoints (x,y,sinθ,cosθ,arrive)（flow matching denoising） |
| **记忆机制** | Learnable latent queries（prompt tuning） | **KV cache** 作为 Central Memory，fast 和 slow 系统共享 |
| **快系统能否独立工作** | 是（System 1 可独立做 point-goal/image-goal/no-goal 导航） | 是（Fast System 可独立做 point-goal / instruct-goal） |
| **慢系统调用频率** | 固定 2Hz | **按需调用**（仅在 ObjectNav 探索或 VLN 路口决策时触发 Slow） |
| **训练方式** | 分阶段训练（先 System 2 SFT，再 System 1，再联合） | 2 阶段联合训练（Stage 1 AR + Stage 2 flow-matching），**始终共享 backbone** |
| **多任务支持** | 仅 VLN（instruct-goal） | **4 种**（point/instruct/object/exploration 统一） |
| **推理频率** | System 2: 2Hz, System 1: 30Hz | Fast: 5Hz, Slow: 按需 |
| **VLM 骨干** | InternVL2.5-7B (7B) | Qwen2.5-VL-3B (3B) |

**核心区别总结**：

1. **共享 vs 分离**：OmniNav 的 fast/slow 共享同一个 VLM backbone（仅 action head 不同），InternVLA-N1 的 System 1 (DiT) 是一个完全独立的模型。OmniNav 更轻量（只需一个 3B VLM），InternVLA-N1 更灵活（System 1 可独立部署）。

2. **Slow 系统的定位**：InternVLA-N1 的 System 2 **每步都运行**（2Hz 固定频率输出 pixel goal），是主控制器。OmniNav 的 Slow System **按需触发**（大部分时间只有 Fast 在跑），是高层规划器——只在需要探索或决策时才调用。

3. **通信方式**：InternVLA-N1 通过 latent goal embedding（连续向量）从 System 2 传到 System 1。OmniNav 通过**坐标文本**（离散的 (x,y) 坐标）从 Slow 传到 Fast，同时通过 KV cache 共享历史上下文。

4. **对 NaviAgent 的启示**：NaviAgent 更接近 InternVLA-N1（分离的 VLM + DiT），但可以借鉴 OmniNav 的"按需调用 Slow System"思路——NaviAgent 的 Latent Adaptive Reasoning（easy 步骤跳过 thinking）本质上就是这个思路。

### 与 NaviAgent 的关联

- **架构高度相关**: OmniNav 的 Fast-Slow 双系统 ≈ NaviAgent 的 System 2 + System 1
- **Flow matching waypoint** ≈ NaviAgent 的 DiT + Spatial Head 输出
- **CoT + frontier selection** ≈ NaviAgent 的 Latent Reasoning + `<tool:mem_r>`
- **Central Memory (KV cache)** ≈ NaviAgent 的 Differentiable Memory Bank（但 NaviAgent 用 learned embeddings 而非 KV cache）
- **通用 VL 数据联训的发现**：NaviAgent 微调时也应混入通用 VL 数据防止遗忘
- **关键区别**: OmniNav 的 fast/slow 共享 VLM（3B 轻量），NaviAgent 的 System 1 DiT 是独立模型。NaviAgent 的 tool tokens 和可微记忆是 OmniNav 没有的

---
