# SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation

> arXiv 2511.21135, 2025.11 | Ziyi Chen*, Yingnan Guo* 等 | 通讯: Zedong Chu† (Amap Alibaba)

- **作者**: Ziyi Chen*, Yingnan Guo*（共一）等 | 通讯: **Zedong Chu†**（Amap Alibaba）
- **发表**: arXiv 2511.21135, 2025.11
- **项目**: https://amap-eai.github.io/SocialNav/

### 做了什么

SocialNav 是一个**社会合规导航基础模型**——不仅要走到目标，还要遵守社会规范（走人行道、不横穿草坪、遵守交通规则）。核心创新是 hierarchical brain-action 架构 + SAFE-GRPO 强化学习。

### 输入/输出

- **Brain Module 输入**: 历史 RGB 帧 + 2D 位置序列 + 目标坐标 + 文本 prompt
- **Brain Module 输出**:
  - 社会可通行区域（polygon 坐标）
  - CoT 推理文本（导航决策解释）
  - VQA 回答（场景理解）
- **Action Expert 输入**: VLM last-layer features (Z_VLM) + 噪声 waypoints
- **Action Expert 输出**: 5-step denoised waypoints（flow matching）

### 核心架构

**Brain Module（VLM）**：
- Qwen2.5-VL-3B
- 生成式文本输出：可通行区域 polygon、CoT 推理、VQA
- 关键能力：**识别哪些区域可以走**（人行道 ✓、草坪 ✗、车道 ✗）

**Action Expert（DiT）**：
- 12 层 Diffusion Transformer，H=12 heads，D=1536
- 条件化于 VLM 的 last-layer features Z_VLM
- Flow matching 生成 5 步 waypoints
- K=5 步 Euler integration 去噪

### 训练数据（SocNav Dataset, 7M 样本）

**Expert Trajectories Pyramid (ETP)**:
| 层级 | 数据量 | 来源 | 说明 |
|------|:---:|------|------|
| D_video | 2.0M | YouTube 城市步行视频 | π³ 3D 重建 + MoGe 尺度对齐 + 伪轨迹采样 |
| D_sim | 1.7M | 4,490 高保真 3D 场景 + SocCity (3.37km²) | 含 SocialGS（3,400 3DGS 场景）和 SocCity 动态城市 |
| D_real | 340K | SCAND, Huron, Recon, CityWalker 真机数据 | 真实物理动力学和传感器噪声 |

**Cognitive Activation Dataset (CAD)**:
| 类型 | 数据量 | 说明 |
|------|:---:|------|
| Social Traversability | 1.2M | 手动标注可通行区域 polygon |
| Navigation CoT | 825K | Qwen2.5-VL-72B 生成推理链 |
| General VQA | 1M | 通用 VL 理解 |

### 训练流程（3 阶段）

1. **Stage 1 Pre-training**: D_video + D_sim + D_cog → 激活导航能力和认知推理
2. **Stage 2 Fine-tuning**: D_real → 仅训 Action Expert（VLM 冻结），缩小 sim-to-real gap
3. **Stage 3 SAFE-GRPO**: 在 SocCity 上做**基于 flow 的 RL**——SDE 替代 ODE，引入可控随机探索。奖励 = $\mathcal{R}_{social}$ + $\mathcal{R}_{expert}$ + $\mathcal{R}_{smooth}$ + $\mathcal{R}_{eff}$

### 关键结果

| Benchmark | 指标 | SocialNav | CityWalker |
|-----------|------|:---:|:---:|
| SocNav Bench (closed-loop) | SR | **86.1%** | 47.8% |
| SocNav Bench | SPL | **77.4%** | 44.7% |
| SocNav Bench | DCR (社会合规) | **82.5%** | 36.1% |
| 真实世界 (3 场景) | SR | **85.0%** | 62.5% |
| CityWalker (open-loop) | MAOE | **7.8** | 15.2 |

### 与 NaviAgent 的关联

- **社会合规**是 NaviAgent 室外导航需要考虑的重要维度（走人行道、不横穿草坪）
- **Brain-Action 架构** ≈ NaviAgent 的 System 2 + System 1
- **SAFE-GRPO** 是 NaviAgent Stage 3 RFT 的直接参考——flow-based RL + 社会奖励
- **SocNav Dataset 的 data pyramid** 启发了 NaviAgent 的分层数据策略（video→sim→real）
- **Traversability prediction** 可对应 NaviAgent 的一种 tool token 功能

---
