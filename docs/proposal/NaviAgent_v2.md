# NaviAgent: A Tool-Augmented Cognitive Navigation Agent for Seamless Indoor-Outdoor Long-Range Exploration

---

## Changelog (v1 → v2)

| #  | 修改                                                                               | 动机                                                                  |
| -- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 1  | 新增 §2.2 POMDP 形式化定义                                                        | 审阅指出缺少形式化，影响 soundness                                    |
| 2  | 新增 §4.4 Adaptive Thinking Strategy（回应 Aux-Think）                            | 审阅发现 Inference-time Reasoning Collapse 风险，这是最核心的技术修改 |
| 3  | 重新设计 `update_map()` 为 Agent 可选调用                                        | 原设计与"Agent 自主决定"矛盾                                          |
| 4  | 新增 §4.10 Algorithm Pseudocode + §4.11 Training Objective                       | 补充形式化，提高 reproducibility                                      |
| 5  | 新增 §7.9 ARNA 对比                                                               | 最直接的竞争对手，v1 遗漏                                             |
| 6  | 新增 §5.4 标注质量保证机制                                                        | 审阅指出标注质量是最大工程风险                                        |
| 7  | 增强 §6 评估：新增 ARNA baseline、Aux-Think 消融、GPS-Threshold Pipeline baseline | 补充关键对比                                                          |
| 8  | 新增 §4.12 计算成本分析                                                           | 审阅指出缺少延迟/效率分析                                             |
| 9  | 扩展 Related Work：ARNA、Aux-Think、VLN-R1、Nav-R1、ETP-R1、TIC-VLA、RAGNav、VAMOS | 文献覆盖不足                                                          |
| 10 | 强化 §11 风险与应对                                                               | 原版 limitation 讨论过于简略                                          |

---

## 1. Motivation

### 1.1 人类是怎么导航的？

给一个人任务"去星巴克买一杯咖啡"，他的大脑会这样运转：

1. **理解任务**：目的地是星巴克→掏出手机→打开高德地图搜索→找到最近的在南边500米外的商场3楼
2. **感知环境**：看看周围——我在办公室，通过窗户看出是在高楼层→判断大概在二楼
3. **制定计划**：需要先下楼→出建筑→跟导航走→进目标商场→上3楼→找星巴克
4. **设定短目标**：当前第一步——找楼梯或电梯下楼
5. **执行**：看到走廊尽头有电梯标志→朝那个方向走
6. **检查进度**：走了一会儿，感觉不对→回想一下刚才的路→发现走重复了→换个方向
7. **到达楼梯**：下楼→确认到了一楼→找到大门出去
8. **切换模式**：到了室外→掏出手机看导航→跟着路线走
9. **持续监控**：走了200米→看一眼手机确认方向对不对→继续走
10. **到达目标建筑**：收起手机→进入商场→找电梯上3楼→靠视觉找星巴克

人类的导航本质上是一个**使用工具的认知Agent**：

- **主动调用工具**：需要的时候掏手机看导航，不需要的时候靠眼睛
- **实时建立心理地图**：记住走过的路、看到的标志物
- **自我监控与纠错**：发现走错了会查看心理地图、重新规划
- **分层决策**：高层规划（从A到B的路线）+ 低层执行（避开眼前的障碍物）
- **自适应推理深度**：简单直行时不需要深度思考，复杂路口需要仔细判断

### 1.2 现有方法的根本缺陷：不是Agent，而是条件反射

现有导航模型的核心问题是——它们是**被动的条件反射式系统**，不是**主动的工具使用Agent**：

**DualVLN / InternVLA-N1**：

- 输入图像→输出pixel goal。没有"思考"过程，不知道自己在干什么
- 不会主动查看记忆（"我是不是走过这里了？"）
- 不会主动调用工具（"我需要看一下地图确认方向"）
- 走错了不会自我纠正——因为没有"走错了"的概念

**UrbanVLA**：

- 被动跟随路线坐标，不会判断路线是否合理
- 进入建筑后GPS漂移，仍然盲目跟随错误路线

**CogNav**：

- 思路最接近——用LLM做认知状态推理+认知地图
- 但是zero-shot LLM调用（每步1-2秒），无法实时部署
- 认知地图很重（Scene Graph + Landmark Graph + Occupancy Map），需要RGB-D和语义分割
- 仅限室内ObjectNav

**ARNA**：

- 思路也接近——用LVLM + tool library做通用导航Agent
- 但基于prompting而非SFT训练，每步需要完整的LVLM API调用（延迟高，无法2Hz部署）
- 工具库是通用的（不针对导航场景设计），缺少导航专用工具（拓扑记忆、回环检测）
- 不涉及跨环境过渡和双系统执行

**BridgeNav**：

- 仅处理"室外→室内"单向过渡，不涉及反向和完整链路

### 1.3 我们的核心思想：把VLM训练成一个会用工具的导航Agent

NaviAgent不是一个"输入→输出"的模型，而是一个**工具增强的认知导航Agent**：

- **它会思考**：在需要时推理当前状态、设定短目标，再决定行动
- **它会建图**：在环境发生变化时构建语义拓扑记忆
- **它会查图**：主动查询记忆来做决策（"我走过这里吗？"、"出口在哪个方向？"）
- **它会用导航**：到了室外主动调用路线API，获取导航指引
- **它会自我纠错**：走了一段时间没有进展→查图→发现绕圈→换方向
- **它会调节推理深度**：简单场景快速反应，复杂场景深度思考

与ARNA（Lange et al., 2025）的关键区别：NaviAgent将Agent行为通过SFT训练内化为模型能力，实现2Hz实时推理（vs ARNA的per-step API调用），并配合轻量扩散策略实现30Hz避障执行。

这不是一个固定pipeline，而是一个**动态的Agent循环**：

```
Observe → [Think?] → [Call Tool?] → Act → Observe → [Think?] → ...
```

注意：`[Think?]` 表示 Agent 自适应地决定是否需要显式推理（Adaptive Thinking，详见 §4.4）。

---

## 2. Problem Definition: Cross-Environment Long-Range Navigation (CELN)

### 2.1 任务定义

给定一个自然语言任务描述（如"去星巴克买一杯咖啡"），机器人从当前室内位置出发，需要自主到达目的地。

机器人配备：

- 多视角RGB相机（始终可用）
- GPS模块（室外可用，室内不可用/不可信）
- 导航工具API（可获取步行路线）

### 2.2 POMDP 形式化定义

CELN 任务建模为一个 Tool-Augmented POMDP：$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{R}, \mathcal{Z}, \gamma \rangle$

**状态空间** $\mathcal{S}$：

- $s_t = (p_t, \theta_t, e_t, f_t)$，其中 $p_t \in \mathbb{R}^2$ 为位置，$\theta_t \in [0, 2\pi)$ 为朝向，$e_t \in \{indoor, outdoor, transition\}$ 为环境类型，$f_t \in \mathbb{Z}^+$ 为楼层

**观测空间** $\mathcal{O}$：

- $o_t = (I_t^{front}, I_t^{left}, I_t^{right}, I_t^{back}, g_t, c_t)$
- $I_t^{v} \in \mathbb{R}^{H \times W \times 3}$：四视角RGB图像
- $g_t \in \mathbb{R}^2 \cup \{\emptyset\}$：GPS坐标（室外可用，室内为$\emptyset$）
- $c_t \in \{high, medium, low, unreliable\}$：GPS置信度（基于HDOP映射）

**动作空间** $\mathcal{A}$：

- 基础动作：$a_t^{nav} = (v, x, y)$，其中 $v \in \{front, left, right, back\}$ 为视角指示，$(x, y)$ 为该视角内的pixel goal坐标
- 工具调用：$a_t^{tool} \in \{update\_map(\cdot), query\_map(\cdot), call\_route(\cdot), check\_progress(), detect\_floor()\} \cup \{\emptyset\}$
- 推理输出：$a_t^{think} \in \mathcal{V}^* \cup \{\emptyset\}$（自然语言 thinking，可选）
- 完整动作：$a_t = (a_t^{think}, a_t^{tool}, a_t^{nav})$

**转移函数** $\mathcal{T}$：$s_{t+1} \sim T(s_{t+1} | s_t, a_t^{nav})$，环境物理动力学

**观测函数** $\mathcal{Z}$：$o_t \sim Z(o_t | s_t)$，包含GPS可用性的随机过程（室内$g_t = \emptyset$，室外$g_t = p_t + \epsilon_t$，$\epsilon_t$ 为GPS噪声）

**奖励** $\mathcal{R}$：

- $R_t = R^{goal}_t + \lambda_1 R^{progress}_t + \lambda_2 R^{efficiency}_t$
- $R^{goal}_t$：到达目标的稀疏奖励
- $R^{progress}_t = \Delta d_t / d_0$：距目标的距离变化（归一化）
- $R^{efficiency}_t = -\mathbb{1}[loop\_detected]$：回环惩罚

**Agent 策略** $\pi_\theta$：

$$
a_t = \pi_\theta(o_t, l, m_t, r_t)
$$

其中 $l$ 为任务指令，$m_t$ 为记忆上下文（上次 `query_map` 结果），$r_t$ 为路线上下文（上次 `call_route` 结果）。策略 $\pi_\theta$ 由 VLM（Qwen3-VL-8B）参数化，输出三元组 $(a_t^{think}, a_t^{tool}, a_t^{nav})$。

### 2.3 与现有任务的区别

| 任务                  | 环境                | 距离                | 导航工具           | 多楼层       | Agent推理           | 代表Benchmark             |
| :-------------------- | ------------------- | ------------------- | ------------------ | ------------ | ------------------- | ------------------------- |
| VLN-CE                | 室内单层            | 10-20m              | 无                 | 否           | 无                  | R2R-CE                    |
| ObjectNav             | 室内                | 5-20m               | 无                 | 部分         | 无/简单             | HM3D                      |
| Urban Nav             | 室外                | 200-2000m           | GPS路线            | N/A          | 无                  | MetaUrban                 |
| Out-to-In Nav         | 室外→室内          | 短距离              | 无                 | 否           | 无                  | BridgeNav                 |
| **CELN (ours)** | **室内+室外** | **200-2000m** | **按需调用** | **是** | **Agent推理** | **NaviAgent-Bench** |

---

## 3. Contribution

1. **NaviAgent：首个SFT-trained工具增强认知导航Agent框架**。将VLM从被动的输入→输出模型，训练为具备工具调用能力的认知Agent。Agent能够：按需建图、主动查图、按需调用导航API、自我监控与纠错。关键技术创新：(a) ReAct-style CoT编码为SFT数据实现端到端可训练（2Hz实时）；(b) Adaptive Thinking策略——训练时用完整CoT内化推理能力，推理时根据场景复杂度动态调节thinking深度（回应Aux-Think的reasoning collapse发现）。与ARNA（prompting-based）不同，NaviAgent的Agent行为是训练内化的，支持实时部署。
2. **CELN任务定义与NaviAgent-Bench评估协议**。定义首个跨环境长程导航任务（POMDP形式化），构建标准化评估体系（逻辑拼接评估 + GPS渐变过渡区评估 + 真机定量评估），设计CELN专用指标（Transition SR、Mode Stability、Reasoning Accuracy等）。
3. **轻量级语义拓扑记忆（作为Agent的按需调用工具）**。融合MapNav的语义地图思想和CogNav的认知地图思想，但以极轻量方式实现（<1MB, <1ms）。记忆模块所有操作（包括建图和查图）均由Agent自主决定何时调用，真正实现按需访问。
4. **开源Baseline + Benchmark + 认知推理标注数据**。包括Agent行为标注pipeline和质量评估指标。

---

## 4. Method

### 4.1 系统架构总览

```
┌─────────────────────────────────────────────────┐
│                  NaviAgent                        │
│                                                   │
│  ┌───────────┐    ┌──────────────────────┐       │
│  │ System 2  │    │    Agent Tools        │       │
│  │ Qwen3-VL  │◄──►│                      │       │
│  │   8B      │    │  Semantic Memory      │       │
│  │           │    │  Route Planner        │       │
│  │ Adaptive  │    │  Progress Checker     │       │
│  │ Reasoning │    │  Floor Detector       │       │
│  │ + Action  │    │                      │       │
│  └─────┬─────┘    └──────────────────────┘       │
│        │                                          │
│        ▼                                          │
│  ┌───────────┐                                    │
│  │ System 1  │    30Hz collision-free trajectory   │
│  │ DiT Policy│──────────────────► Low-level Ctrl  │
│  └───────────┘                                    │
└─────────────────────────────────────────────────┘
```

**System 2（Qwen3-VL-8B，2Hz）**：认知Agent核心——自适应推理、决策、工具调用、pixel goal预测

**Agent Tools**：Agent按需调用的工具集

- Semantic Memory（语义拓扑记忆）：建图/查图/回环检测
- Route Planner（路线规划）：调用导航API获取路线
- Progress Checker（进度检查）：判断是否在前进/是否绕圈
- Floor Detector（楼层检测）：判断当前楼层

**System 1（DiT扩散策略，30Hz）**：低层执行——接收latent goal，生成避障轨迹

### 4.2 Agent的工具定义

NaviAgent的VLM被训练为一个具备工具调用能力的Agent。**所有工具调用均由Agent自主决定**（包括update_map）：

| 工具                 | 触发条件（Agent自主决定）                                     | 输入                              | 输出（注入prompt）                                                                    |
| -------------------- | ------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------- |
| `update_map()`     | Agent判断环境发生变化时（如进入新房间、转弯后、场景类型切换） | 当前位置估计+VLM视觉特征+语义标签 | — (写入记忆)                                                                         |
| `query_map()`      | Agent主动决定查图（感觉走过/需要进度信息/需要楼层信息）       | 查询类型（轨迹/回环/进度/楼层）   | 文本（轨迹摘要/回环警告/进度报告/楼层信息）                                           |
| `call_route()`     | Agent判断GPS可用且需要路线时                                  | 当前GPS位置+目的地                | 路线waypoints（前方40m，20个egocentric航点，编码为相对极坐标序列）+转弯指令+GPS置信度 |
| `check_progress()` | Agent感觉长时间未进展时                                       | 最近N步的位置序列                 | 前进距离/绕圈检测/效率评分                                                            |
| `detect_floor()`   | 检测到楼梯/电梯/楼层变化时                                    | 当前视觉观测                      | 当前楼层估计                                                                          |

**关于 `update_map()` 的设计决策**（v2修改）：

v1中 `update_map()` 被设计为每步自动调用，这与"Agent自主决定工具调用"的核心卖点矛盾。v2修改为Agent自主决定：

- **高频更新场景**：进入新房间、穿过门、转弯后、环境类型切换（室内→室外）
- **低频/跳过场景**：长走廊直行、开阔室外直行（VLM特征变化率低，更新无新增信息）
- **训练标注**：通过VLM特征变化率（$\Delta f_t = \|h_t - h_{t-k}\|_2$）和语义标签变化来标注何时应/不应调用 `update_map()`

这使得Agent可以学到更高效的建图策略：变化大时密集更新，变化小时节省计算。

### 4.3 Agent的推理-行动循环

每一步（2Hz）的完整流程（含Adaptive Thinking）：

```
输入：
  [Observation] 四视角RGB
  [Task] "去星巴克买一杯咖啡"
  [Memory Context] (上次query_map的结果，如果有的话)
  [Route Context] (上次call_route的结果，如果有的话)

Agent输出（Adaptive Thinking模式）：

--- 场景A：复杂场景（需要深度思考） ---

  [Thinking]
  I'm at a T-junction on floor 2. Left leads to a corridor I haven't
  explored. Right has an elevator sign. My goal is to go down to floor 1.
  I should take the right path toward the elevator.

  [Tool Call]
  update_map(scene_type="junction")
  detect_floor()

  [Action]
  Pixel Goal: (right, 156, 312)

--- 场景B：简单场景（快速反应） ---

  [Action]
  Pixel Goal: (front, 320, 180)

--- 场景C：纠错场景（查图+深度思考） ---

  [Thinking]
  This corridor looks familiar. I've been walking without progress.

  [Tool Call]
  query_map(type="loop_check")
  → Response: "WARNING: You visited this area 2 minutes ago.
     Trajectory: office→corridor→turn_right→corridor(HERE, REVISITED).
     Suggest: try going left instead of right at the last junction."

  [Thinking]
  Loop confirmed. Turning around to try the left path.

  [Action]
  Pixel Goal: (back, 320, 240)
```

### 4.4 Adaptive Thinking Strategy（v2新增，核心技术创新）

**问题**：Aux-Think（2025）发现在VLN中，推理时让模型输出CoT反而可能降低导航精度（"Inference-time Reasoning Collapse"）——错误的推理链导致错误的行动决策。

**分析**：NaviAgent的thinking与标准VLN CoT有本质区别：

- 标准VLN CoT：纯推理链（"我应该往左因为指令说..."），错误推理直接导致错误动作
- NaviAgent thinking：**功能性推理**，驱动工具调用决策（"环境变化了→update_map"、"感觉绕圈→query_map"）。工具调用的返回结果提供**外部验证信号**，部分纠正推理错误

但风险仍然存在。因此我们提出**Adaptive Thinking**——三种策略的统一框架：

| 策略                             | 训练时              | 推理时                | 适用场景                           |
| -------------------------------- | ------------------- | --------------------- | ---------------------------------- |
| **Always-Think**           | 完整CoT             | 完整CoT               | Baseline / 分析实验                |
| **Think-to-Internalize**   | 完整CoT（辅助loss） | 无CoT，直接action     | 借鉴Aux-Think，内化推理为隐式能力  |
| **Adaptive-Think（推荐）** | 完整CoT + 难度标签  | Agent决定是否thinking | 简单场景快速反应，复杂场景深度推理 |

**Adaptive-Think实现**：

训练时，对每步标注场景难度 $d_t \in \{easy, medium, hard\}$：

- `easy`：长走廊直行、开阔室外跟随路线 → 标注仅 `[Action]`
- `medium`：普通导航决策 → 标注短 `[Thinking]` + `[Action]`
- `hard`：路口决策、环境切换、纠错 → 标注完整 `[Thinking]` + `[Tool Call]` + `[Action]`

推理时，VLM学会根据场景复杂度自适应输出：

- 简单场景：直接预测 `[Action]`（无thinking overhead，保持2Hz）
- 复杂场景：输出 `[Thinking]` + 可选 `[Tool Call]` + `[Action]`

**难度标注规则**：

- `hard`：GPS状态变化（$c_t \neq c_{t-1}$）、语义标签变化（新场景类型）、回环检测触发、楼层变化
- `medium`：路口/转弯、子目标切换
- `easy`：其余

### 4.5 认知状态（Agent自主判断，不靠硬编码规则）

Agent通过推理自主判断当前状态，而非由GPS阈值等规则触发：

| 状态                    | Agent的典型推理                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| Indoor Exploration      | "I'm inside a building, no GPS. Looking for stairs or exit."                              |
| Floor Transition        | "I found the elevator. Going down to floor 1."                                            |
| Exit Seeking            | "I'm on floor 1 now. The route starts from the south, so I'm looking for the south exit." |
| Outdoor Route Following | "I'm outside, GPS is good. Following the navigation route."                               |
| Building Entry          | "I can see the target building. Looking for an entrance."                                 |
| Target Seeking          | "I'm inside the target building. Looking for Starbucks on floor 3."                       |
| Self-Correction         | "I seem to be going in circles. Let me check my map and change direction."                |

**与CogNav/ARNA的关键区别**：

| 维度     | CogNav                    | ARNA                 | NaviAgent                     |
| -------- | ------------------------- | -------------------- | ----------------------------- |
| 推理方式 | Zero-shot LLM (1-2s/step) | Prompting-based LVLM | SFT-trained VLM (2Hz)         |
| 工具调用 | 固定频率                  | Per-step API call    | Agent按需 + Adaptive Thinking |
| 环境范围 | 室内单层 ObjectNav        | 室内通用导航         | 跨环境 500m+                  |
| 执行方式 | 单系统                    | 单系统               | 双系统 (2Hz + 30Hz)           |
| 部署性   | 不可实时                  | API依赖              | 实时可部署                    |

### 4.6 语义拓扑记忆（Agent的按需调用工具）

**设计原则**：所有操作（含建图）由Agent自主决定何时调用。

**数据结构**：增量式语义拓扑图

- 节点创建条件：Agent调用 `update_map()`且距上一节点 > 2m
- 节点存储：位置估计（室外GPS / 室内里程计累积）、256维VLM视觉特征（VLM倒数第4层pooled hidden state，零额外计算）、语义标签（Agent在thinking中推断的场景类型）、楼层标签
- 边存储：节点间的连接关系和距离

**工具接口**：

`update_map(scene_type)`：Agent决定在环境变化时调用

- 输入：当前位置估计、VLM视觉特征、场景类型
- 操作：创建新节点（如距离上一节点>2m）或更新当前节点
- 不调用的条件：Agent判断环境无变化（如直行走廊中）

`query_map(type)`：Agent主动决定何时调用

- `type="trajectory"` → 返回轨迹语义序列（"office→corridor→elevator(↓)→lobby"）
- `type="loop_check"` → 回环检测（余弦相似度 + 语义标签双重验证 + 方向一致性检查），返回是否绕圈及建议。**注**：为缓解语义相似但位置不同的误判（如办公楼多段相似走廊），增加位置距离约束：仅当视觉相似度 > 0.9 **且** 拓扑距离 > 10个节点时才触发回环警告
- `type="progress"` → 计算已行进距离、估计剩余距离、效率评分
- `type="floor_info"` → 返回当前楼层和楼层历史

**轻量性**：500米路线约50-150个节点（Agent按需建图比固定频率更少），<500KB内存，回环检测<1ms。

**与现有方法的对比**：

| 维度     | CogNav认知地图       | MapNav ASM                | RAGNav          | NaviAgent语义拓扑记忆       |
| -------- | -------------------- | ------------------------- | --------------- | --------------------------- |
| 组成     | SG+LG+OccMap         | 多通道2D语义地图+文本标注 | 拓扑图+语义森林 | 单一拓扑图                  |
| 输入     | RGB-D+open-vocab seg | RGB-D+语义分割            | RGB+depth(可选) | **仅RGB**             |
| 建图频率 | 固定频率             | 每步更新                  | 每步更新        | **Agent按需**         |
| 查询方式 | 固定频率             | 固定                      | RAG检索         | **Agent按需**         |
| 计算     | 重(3D点云,DBSCAN)    | 中(点云投影)              | 中(检索+排序)   | **极轻(<1ms)**        |
| 范围     | 室内单层             | 室内单层                  | 室内            | **室内外跨楼层500m+** |

### 4.7 Route Planner工具

Agent到了室外决定调用导航时：

`call_route(destination)`：

- 调用高德步行路线API
- 截取前方40m片段，重采样为20个egocentric航点
- 航点编码：相对极坐标序列 $(r_i, \alpha_i)$，$r_i$ 为距离（米），$\alpha_i$ 为相对当前朝向的角度
- 返回：waypoints + 转弯指令（自然语言） + GPS置信度标签

**GPS置信度**：HDOP映射为离散标签

- $HDOP \leq 4$：high → waypoints可信
- $4 < HDOP \leq 8$：medium → 参考但需视觉确认
- $8 < HDOP \leq 15$：low → 仅做方向参考
- $HDOP > 15$ 或无信号：unreliable → 不使用

**Agent自主决策模式切换**（vs 硬编码GPS阈值）：

- GPS信号一般但能看到目标→不调route，直接走
- GPS信号好但route指向一堵墙→忽略route，靠视觉
- GPS信号从bad变good→先试探性调route，确认合理后再跟随

### 4.8 VLM选型：Qwen3-VL-8B

选择理由：

- 256K context：四相机 + 历史 + route + memory + reasoning 全部可容纳
- DeepStack多层特征融合：视觉理解质量
- 原生支持交错文本-图像输入：适合多轮 tool response 注入

训练方案基于InternVLA-N1迁移：LR=1e-5，Vision LR=2e-6，BF16，8×A100。

### 4.9 System 1：DiT扩散策略

复用DualVLN设计（TIC-VLA验证了异步VLM规划+低层控制的可行性），接收System 2的latent goal + 30Hz实时RGB生成避障轨迹。与Agent的工具调用无关。

### 4.10 算法伪代码（v2新增）

```
Algorithm: NaviAgent Inference Loop

Input: task instruction l, initial observation o_0
Initialize: memory M = {}, route_context r = null, memory_context m = null

for t = 0, 1, 2, ... (at 2Hz):
    # Encode observation
    h_t = VLM.encode(o_t)  # VLM visual features

    # Construct prompt
    prompt = format_prompt(o_t, l, m, r)

    # VLM generates (think, tool_call, action) autoregressively
    output = VLM.generate(prompt)

    # Parse output (Adaptive Thinking: output may or may not contain thinking)
    think_t, tool_calls_t, action_t = parse(output)

    # Execute tool calls (if any)
    for tool_call in tool_calls_t:
        if tool_call.name == "update_map":
            M.add_node(position_estimate(o_t), h_t, tool_call.args.scene_type)
        elif tool_call.name == "query_map":
            m = M.query(tool_call.args.type)
            # Re-inject response and continue generation
            output = VLM.continue_generate(tool_response=m)
            think_t2, _, action_t = parse(output)  # Post-tool thinking + action
        elif tool_call.name == "call_route":
            r = RouteAPI.query(o_t.gps, tool_call.args.destination)
            output = VLM.continue_generate(tool_response=r)
            think_t2, _, action_t = parse(output)
        elif tool_call.name == "check_progress":
            progress = M.compute_progress()
            output = VLM.continue_generate(tool_response=progress)
            think_t2, _, action_t = parse(output)
        elif tool_call.name == "detect_floor":
            floor = M.estimate_floor(h_t)
            output = VLM.continue_generate(tool_response=floor)
            think_t2, _, action_t = parse(output)

    # Send pixel goal to System 1
    latent_goal = VLM.project_goal(action_t)  # Project to latent space
    System1.set_goal(latent_goal)  # DiT generates 30Hz trajectory

    # Check termination
    if action_t == STOP: break
```

### 4.11 训练目标（v2新增）

**SFT阶段**：标准 next-token prediction，但对不同部分使用差异化权重：

$$
\mathcal{L}_{SFT} = -\sum_t \left[ w_{think} \cdot \log p(a_t^{think}) + w_{tool} \cdot \log p(a_t^{tool}) + w_{nav} \cdot \log p(a_t^{nav}) \right]
$$

- $w_{think} = 0.3$：thinking文本（辅助信号，不是核心目标）
- $w_{tool} = 0.5$：工具调用决策（关键——学会何时调用什么工具）
- $w_{nav} = 1.0$：pixel goal（导航行为的核心）

**Adaptive Thinking的loss处理**：

- `easy` 样本：仅有 $w_{nav}$ 项（无thinking/tool标注）
- `medium` 样本：$w_{think}$ + $w_{nav}$
- `hard` 样本：全部三项

**Think-to-Internalize消融**（借鉴Aux-Think）：

- 训练时：$w_{think} > 0$（完整CoT辅助loss）
- 推理时：$w_{think} = 0$（不输出thinking，直接action）
- 实现：推理时用 `[Action]` 作为 forced prefix

**RFT阶段**：

Plan A：IQL（沿用UrbanVLA验证方案）

$$
\mathcal{L}_{IQL} = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ L_\tau^2(Q(s,a) - V(s)) + (r + \gamma V(s') - Q(s,a))^2 \right]
$$

Plan B：GRPO（沿用VLN-R1/ETP-R1验证方案）

$$
\mathcal{L}_{GRPO} = -\mathbb{E} \left[ \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A(s,a) - \beta \cdot KL(\pi_\theta || \pi_{ref}) \right]
$$

其中 advantage $A(s,a)$ 使用 Time-Decayed Reward（沿用VLN-R1）。

### 4.12 计算成本分析（v2新增）

| 方法                            | 推理频率      | tokens/step (输入)               | tokens/step (输出)                    | 估计延迟         | GPU内存         |
| ------------------------------- | ------------- | -------------------------------- | ------------------------------------- | ---------------- | --------------- |
| DualVLN                         | 2Hz           | ~2000 (4 imgs)                   | ~20 (pixel goal)                      | ~200ms           | ~16GB           |
| CogNav (GPT-4V)                 | ~0.5Hz        | ~3000 (img+map)                  | ~200 (reasoning)                      | 1-2s (API)       | N/A (API)       |
| ARNA (prompting)                | ~0.5Hz        | ~5000 (img+tools+history)        | ~300 (reasoning+tools)                | 1-3s (API)       | N/A (API)       |
| **NaviAgent (easy step)** | **2Hz** | **~2500**                  | **~20 (action only)**           | **~250ms** | **~20GB** |
| **NaviAgent (hard step)** | **2Hz** | **~3000 (+ memory/route)** | **~80-120 (think+tool+action)** | **~400ms** | **~20GB** |
| NaviAgent (weighted avg)        | 2Hz           | ~2600                            | ~40                                   | ~280ms           | ~20GB           |

**分析**：

- Adaptive Thinking下，约60%的步骤为easy/medium（无/短thinking），40%为hard（完整thinking+tool call）
- 加权平均推理延迟约280ms/step，满足2Hz要求（500ms budget）
- 相比DualVLN增加约40%延迟，但获得Agent推理能力
- 相比CogNav/ARNA快3-7倍，且不依赖API

---

## 5. Training Data

### 5.1 SFT数据构成

| 数据类型    | 来源                           | 规模            | 训练能力           |
| ----------- | ------------------------------ | --------------- | ------------------ |
| 室内VLN     | R2R-CE + ScaleVLN(2万子集)     | ~30K episodes   | Indoor Exploration |
| 室内找出口  | HM3D出口标注（自构造）         | ~4K episodes    | Exit Seeking       |
| 室外Route   | MetaUrban PPO Expert           | ~5-10K episodes | Route Following    |
| 过渡区数据  | Habitat内GPS渐变模拟（自构造） | ~1-2K episodes  | Mode Switching     |
| GPS噪声增强 | 上述所有×4                    | ×4扩增         | GPS Robustness     |

### 5.2 Agent行为标注（核心数据创新）

**这是NaviAgent区别于所有现有工作的关键训练数据。**

对每条轨迹的每一步，标注完整的Agent行为：(Thinking + Tool Call + Action) 或 (Action)，取决于场景难度。

**自动标注pipeline**：

1. **场景难度标注**（v2新增）：

   - `hard`：GPS状态变化、语义标签变化、楼层变化、VLM特征变化率 > 阈值$\tau_h$
   - `medium`：路口/转弯（检测方向变化 > 30°）、子目标切换
   - `easy`：其余（直行、稳定跟随路线）
   - 目标分布：~60% easy, ~25% medium, ~15% hard
2. **状态标注**：基于规则（GPS值→室内/室外、高度变化→楼层、距出口距离→Exit Seeking等）
3. **Thinking生成**：

   - 仅对 `medium` 和 `hard` 步骤生成
   - Level 0（规则模板）：基础版（"I'm in {scene_type} on floor {floor}. My goal is {goal}."）
   - Level 1（VLM增强）：用Qwen3-VL-8B-Instruct对当前帧做场景描述，生成更自然的描述
   - Level 2（人工修正）：对随机10%的Level 1标注进行人工审核和修正
4. **Tool Call标注**：

   - `update_map()`：Agent判断环境变化时（VLM特征变化率 > $\tau_u$，或语义标签变化）
   - `query_map()`：
     - 走了20步以上未到达子目标→标注query_map(type="progress")
     - 当前视觉特征与历史节点相似 **且** 拓扑距离 > 10节点→标注query_map(type="loop_check")
     - 到达新楼层→标注query_map(type="floor_info")
   - `call_route()`：GPS从unreliable变为medium/high时→标注call_route()
   - `check_progress()`：每30步自动标注一次
   - `detect_floor()`：检测到高度显著变化时
5. **Action标注**：pixel goal坐标（沿用InternVLA-N1的标注方式）

**示例训练样本**：

**Easy步骤（占~60%）**：

```json
{
  "observation": "<front_img> <left_img> <right_img> <back_img>",
  "task": "Go to Starbucks in Building B",
  "difficulty": "easy",
  "output": "pixel_goal: (front, 320, 180)"
}
```

**Hard步骤——过渡区（占~15%）**：

```json
{
  "observation": "<front_img showing glass door and outdoor view>",
  "task": "Go to Starbucks in Building B",
  "difficulty": "hard",
  "memory_context": "Trajectory: office→corridor→elevator(↓)→lobby\nFloor: 1\nProgress: 45m/500m",
  "route_context": "unavailable",
  "output": "[Thinking] I can see the glass entrance door ahead with sunlight outside. I'm about to exit the building. Once outside, I should try to get navigation route.\n[Tool Call] update_map(scene_type='lobby_exit')\ncall_route(destination='Starbucks, Building B')\n[Action] pixel_goal: (front, 320, 250)"
}
```

**Hard步骤——自我纠错**：

```json
{
  "observation": "<front_img showing a familiar-looking corridor>",
  "task": "Go to Starbucks in Building B",
  "difficulty": "hard",
  "memory_context": "Trajectory: ...→corridor→right_turn→corridor\nProgress: stagnant for 8 steps",
  "output": "[Thinking] This corridor looks familiar. Progress has been stagnant.\n[Tool Call] query_map(type='loop_check')\n[Tool Response] LOOP DETECTED: You visited this area 45 seconds ago. Last junction was 3 steps back. Suggest: turn left.\n[Thinking] Loop confirmed. Going back to try the left path.\n[Action] pixel_goal: (back, 320, 240)"
}
```

### 5.3 过渡区数据构造

在Habitat长轨迹上模拟GPS过渡：

- **类型A（出建筑）**：GPS=unreliable → 渐变 → GPS=high + route出现
- **类型B（进建筑）**：反向
- **类型C（GPS波动）**：medium/low交替
- **类型D（突变）**：GPS直接跳变

过渡段的Agent行为标注重点：Agent何时决定调用 `call_route()`、何时决定停止信任route。

### 5.4 标注质量保证机制（v2新增）

| 质量维度        | 指标                                   | 评估方法                                  | 目标                            |
| --------------- | -------------------------------------- | ----------------------------------------- | ------------------------------- |
| Thinking多样性  | Distinct-4 (4-gram diversity)          | 计算所有thinking文本的distinct 4-gram比例 | > 0.7                           |
| Tool Call合理性 | 人工评判precision/recall               | 200个随机episode × 人工标注（3人投票）   | Precision > 0.85, Recall > 0.75 |
| 标注一致性      | Inter-annotator agreement (Cohen's κ) | 50个episode由3人独立标注tool call时机     | κ > 0.7                        |
| OOD泛化性       | 在未见场景上的tool call precision      | 用训练好的模型在HM3D holdout场景上测试    | 与训练场景差距 < 10%            |

**分级标注流程**：

1. Level 0（全部数据）：规则模板生成 → 快速覆盖
2. Level 1（全部数据）：VLM captioning增强 → 提升自然度和多样性
3. Level 2（10%随机采样）：人工审核和修正 → 质量校准
4. 质量检查：每1000条计算Distinct-4和一致性，低于阈值则回溯修正

---

## 6. Evaluation

### 6.1 NaviAgent-Bench 评估体系

**第一层：逻辑拼接评估（E2E能力）**

- Indoor段（Habitat+HM3D）+ Outdoor段（MetaUrban），1000-2000 episodes

**第二层：GPS渐变过渡区评估（切换能力）**

- 50个HM3D半室外场景，200 episodes

**第三层：GPS噪声鲁棒性（鲁棒性）**

- 固定100 episodes，σ={2,5,10,20,50}m

### 6.2 Baseline对比（v2增强）

| 方法                                     | 类型                          | Agent推理               | 工具调用               | 新增标记         |
| ---------------------------------------- | ----------------------------- | ----------------------- | ---------------------- | ---------------- |
| DualVLN Only                             | 纯VLN                         | 无                      | 无                     |                  |
| UrbanVLA Only                            | 纯Route                       | 无                      | 固定GPS                |                  |
| **GPS-Threshold Pipeline**         | **组合+HDOP滞回**       | **无**            | **规则切换**     | **v2新增** |
| CogNav (adapted)                         | 认知+zero-shot LLM            | 有(LLM)                 | 固定频率               |                  |
| **ARNA (adapted)**                 | **认知+prompting LVLM** | **有(prompting)** | **per-step API** | **v2新增** |
| Oracle Switch                            | 组合+Oracle切换               | 无                      | Oracle                 |                  |
| **NaviAgent**                      | **认知+端到端Agent**    | **有(SFT训练)**   | **Agent按需**    |                  |
| NaviAgent w/o Thinking                   | 消融                          | 无                      | 按需                   |                  |
| **NaviAgent Think-to-Internalize** | **消融**                | **训练有/推理无** | **按需**         | **v2新增** |
| NaviAgent w/o Tools                      | 消融                          | 有                      | 无                     |                  |
| NaviAgent w/o Memory                     | 消融                          | 有                      | 无记忆                 |                  |
| **NaviAgent Always-Think**         | **消融**                | **每步thinking**  | **Agent按需**    | **v2新增** |

**关键对比逻辑**：

- **GPS-Threshold Pipeline vs NaviAgent**：验证Agent推理是否比简单规则切换更好（最关键的 baseline）
- **ARNA vs NaviAgent**：SFT-trained vs prompting-based tool agent（延迟和精度的 trade-off）
- **Always-Think vs Adaptive-Think vs Think-to-Internalize**：回应Aux-Think的reasoning collapse（核心消融）
- **NaviAgent w/o Memory**：记忆模块的实际价值
- **NaviAgent w/o Tools**：工具调用 vs 纯推理（是工具有用还是thinking有用？）

### 6.3 实验列表（v2增强）

| 实验                                    | 目的                                                       | v2变化                                    |
| --------------------------------------- | ---------------------------------------------------------- | ----------------------------------------- |
| 实验一：室内VLN                         | 验证不退化（R2R-CE, RxR-CE）                               |                                           |
| 实验二：室外Route Following             | 验证Route能力（MetaUrban）                                 |                                           |
| 实验三：CELN主实验                      | 核心E2E评估                                                | 新增GPS-Threshold Pipeline和ARNA baseline |
| 实验四：GPS噪声鲁棒性                   | SR vs σ曲线                                               |                                           |
| 实验五：RFT消融                         | IQL/GRPO有效性                                             |                                           |
| 实验六：Agent行为消融                   | Thinking/Tool/Memory各自贡献                               |                                           |
| **实验七：Adaptive Thinking消融** | **Always-Think vs Adaptive vs Think-to-Internalize** | **v2新增，核心**                    |
| 实验八：工具调用分析                    | Agent何时调用哪个工具？调用频率？准确性？OOD泛化？         | 增加OOD场景分析                           |
| 实验九：自我纠错分析                    | Agent检测到多少次绕圈？纠错成功率？误报率？                | 增加误报率分析                            |
| **实验十：标注质量消融**          | **Level 0 vs Level 1 vs Level 2标注的训练效果**      | **v2新增**                          |
| 实验十一：真机定量评估                  | 校园10-15条路线                                            |                                           |

### 6.4 关键指标

标准指标：SR、SPL、NE、nDTW

CELN专用指标：

- E2E SR / E2E SPL
- Transition SR / Transition Reaction Time / Mode Stability
- GPS Noise Robustness曲线 / Route Reliance曲线

Agent专用指标：

- **Reasoning Accuracy**：认知状态判断准确率
- **Tool Call Precision/Recall**：工具调用的合理性（不该调的时候调了=false positive，该调的时候没调=false negative）
- **Self-Correction Rate**：自我纠错的成功率
- **Loop Detection Rate / False Positive Rate**：绕圈检测率和误报率
- **Thinking Efficiency**：Adaptive Thinking的thinking触发比例 vs 导航性能（理想：thinking比例低但性能不降）
- **Inference Latency**：实际推理延迟分布（ms/step）

---

## 7. Related Work

### 7.1 Vision-Language Navigation

DualVLN (2025)、InternVLA-N1 (2025)、JanusVLN (2025)、StreamVLN (2025)——室内VLN SOTA，但是被动的输入→输出模式，无Agent推理。

### 7.2 室外导航与Route Following

UrbanVLA (2025)、UrbanNav (AAAI 2026)、CityWalker (CVPR 2025)——室外导航，不处理室内和过渡。

### 7.3 认知推理导航

CogNav (Cao et al., 2025)——LLM驱动的认知状态机+异构认知地图。ObjectNav SOTA（HM3D 72.5%）。但zero-shot、仅室内单层、计算重。

### 7.4 Chain-of-Thought与推理策略导航

NavCoT (AAAI 2024)——VLN的CoT推理。**Aux-Think (2025)**——首次系统评估VLN推理策略，发现Inference-time Reasoning Collapse，提出训练时CoT辅助监督+推理时直接预测。EvolveNav (2025)——自改进推理范式。**NaviAgent的Adaptive Thinking是对Aux-Think发现的直接回应**：通过场景难度驱动的自适应推理深度，在reasoning collapse风险和推理能力之间取得平衡。

### 7.5 VLN强化微调

**VLN-R1 (2025)**——首个GRPO-based RFT用于端到端VLN，Time-Decayed Reward。**ETP-R1 (2025)**——graph-based VLN-CE上GRPO RFT SOTA。**Nav-R1 (2025)**——Fast-in-Slow reasoning + GRPO，与NaviAgent双系统理念一致。MobileVLA-R1 (2025)——四足机器人CoT+GRPO。这些工作验证了NaviAgent的SFT→RFT训练路线可行。

### 7.6 语义地图与空间记忆

MapNav (Zhang et al., 2025)——ASM替代历史帧。**RAGNav (2026)**——双基底记忆系统（拓扑图+语义森林）。**TagaVLM (2026)**——拓扑结构注入VLM backbone。VoroNav——Voronoi图+LLM接口。INHerit-SG——层次化场景图。MemoNav (CVPR 2024)——工作记忆模型。**NaviAgent的记忆是Agent的工具，按需调用而非固定更新。**

### 7.7 多楼层导航

ASCENT (Gong et al., 2026)——多楼层零样本ObjectNav。

### 7.8 室内外过渡

BridgeNav (Zhao et al., 2026)——仅out-to-in单向过渡。**VAMOS (2025)**——跨室内外VLM导航，VLM pixel path + affordance model，但无工具调用和认知推理。

### 7.9 工具增强导航Agent（v2新增）

**ARNA (Lange et al., 2025)**——最相关竞争对手。LVLM + tool library做通用导航Agent，Agent自主定义和执行task-specific workflows。但基于prompting（每步完整API调用，延迟1-3秒），工具库为通用设计（缺少导航专用的拓扑记忆、回环检测），仅室内单环境，无双系统执行。**NaviAgent与ARNA的核心区别**：(1) SFT-trained Agent行为，2Hz实时部署 vs prompting-based API依赖；(2) 导航场景专用工具设计（拓扑记忆+回环检测+GPS-aware route planner）；(3) 跨环境（室内+室外+过渡）vs 室内；(4) Dual-system execution（VLM 2Hz + DiT 30Hz）vs 单系统。

**CoINS (2026)**——skill-aware VLM + RL技能库，VLM决定何时调用哪个技能。概念上与NaviAgent的"VLM决定何时调用哪个工具"类似，但应用于interactive navigation而非长程跨环境导航。

### 7.10 双系统与异步推理

**TIC-VLA (Huang et al., 2026)**——显式建模VLM推理延迟与反应控制的异步问题，cached hidden states + latency metadata。Fast-in-Slow (2025)——参数共享的双系统VLA。**NaviAgent的双系统设计与TIC-VLA的延迟建模是互补的**。

### 7.11 导航基础模型与Agent

NavFoM (Zhang et al., 2025)——跨任务跨平台导航基础模型。NaVILA (RSS 2025)——VLA生成mid-level语言动作+locomotion RL执行。ReAct (Yao et al., 2023)——推理+行动的通用Agent范式。**NaviAgent将ReAct范式首次通过SFT训练应用于具身导航Agent。**

---

## 8. Novelty分析

### 8.1 NaviAgent的核心创新

**不是DualVLN + UrbanVLA的拼凑，不是ARNA的导航版——而是一个新的训练范式：将Agent行为内化为VLM能力。**

| 维度     | 现有模型范式 | ARNA（prompting Agent）       | NaviAgent（SFT-trained Agent）                  |
| -------- | ------------ | ----------------------------- | ----------------------------------------------- |
| 决策     | 输入→输出   | 思考→工具→行动（prompting） | 思考→工具→行动（训练内化）                    |
| 推理深度 | 固定         | 固定（每步完整推理）          | **自适应**（Adaptive Thinking）           |
| 部署延迟 | 低           | 高（API调用）                 | **低**（SFT-trained, 2Hz）                |
| 工具设计 | 无           | 通用工具库                    | **导航专用**（拓扑记忆+回环检测+GPS路线） |
| 环境范围 | 单环境       | 单环境（室内）                | **跨环境**（室内+室外+过渡）              |
| 执行方式 | 单系统       | 单系统                        | **双系统**（2Hz + 30Hz）                  |
| 记忆管理 | 固定频率/无  | 每步更新                      | **Agent按需**                             |
| 纠错能力 | 无           | 有（prompting）               | 有（**训练内化** + 回环检测工具）         |

### 8.2 审稿人可能的质疑与应对

| 质疑                                                           | 应对                                                                                                                                                                                                                                                      |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "Agent只是在SFT数据中加了Thinking和Tool Call文本，本质还是SFT" | 训练方式确实是SFT，但(1)数据格式的改变带来行为模式的改变——Adaptive Thinking让模型学会何时需要推理；(2)工具调用产生外部反馈（map query结果），形成闭环而非开环预测；(3)实验七八九的消融和分析证明这些行为是真实学到的（重点：OOD场景的工具调用精度分析） |
| "Aux-Think发现CoT会降低VLN性能，NaviAgent的Thinking是否也会？" | 直接在实验七中消融三种策略（Always-Think / Adaptive-Think / Think-to-Internalize）。NaviAgent的Thinking与标准VLN CoT的区别在于：(1)功能性（驱动工具调用）而非仅推理链；(2)自适应（简单场景不thinking）；(3)有外部验证（工具返回结果部分纠正推理错误）     |
| "ARNA已经做了tool-augmented navigation agent"                  | ARNA是prompting-based（延迟高、不可实时）、室内-only、无双系统执行。NaviAgent的核心差异化是将Agent行为通过SFT训练内化为模型能力，实现2Hz实时部署。实验将直接对比两者的延迟和精度trade-off                                                                 |
| "工具调用是否增加了推理延迟"                                   | 提供计算成本分析（§4.12）。Adaptive Thinking下，60%步骤为easy（无thinking overhead），加权平均280ms/step满足2Hz要求。工具本身极轻量（<1ms）                                                                                                              |
| "自我纠错在训练数据中如何标注？"                               | 在训练数据中故意引入"走错→查图→发现→纠正"的轨迹（基于正确轨迹人工添加偏航段）。纠错样本占比约5%，通过实验九验证纠错能力和对正常导航的影响                                                                                                              |
| "逻辑拼接benchmark的限制"                                      | 诚实承认；GPS渐变评估+真机实验弥补。同时设计GPS-Threshold Pipeline baseline来验证过渡处理的重要性                                                                                                                                                         |

---

## 9. Timeline（16周）

| 阶段             | 周    | 任务                                                                                                                      |
| ---------------- | ----- | ------------------------------------------------------------------------------------------------------------------------- |
| Phase 1 数据     | 1-4   | 环境搭建、数据下载/渲染、出口标注、MetaUrban收集、**Agent行为标注pipeline开发**（含难度标注、质量检查）、过渡区数据 |
| Phase 2 SFT      | 5-7   | Qwen3-VL-8B SFT（含Agent行为数据+Adaptive Thinking）、Benchmark构建、实验一二                                             |
| Phase 3 RFT+实验 | 8-10  | Sim-RFT(IQL/GRPO)、实验三~十                                                                                              |
| Phase 4 真机     | 11-13 | 遥操作+Real-RFT、真机10-15条路线                                                                                          |
| Phase 5 写作     | 14-16 | 论文+开源准备                                                                                                             |

---

## 10. 投稿策略

**论文定位**：Method + Benchmark双贡献

- Method：首个SFT-trained工具增强认知导航Agent（从"模型"到"Agent"的范式转变，含Adaptive Thinking策略）
- Benchmark：CELN任务+NaviAgent-Bench

**推荐venue**：

- CoRL 2026（Robotics+Agent，非常合适）
- NeurIPS 2026（Agent+Embodied AI track）
- RSS 2026
- RA-L

---

## 11. 风险与应对（v2增强）

| 风险                                                   | 严重度       | 应对                                                                                                                                     |
| ------------------------------------------------------ | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Agent行为标注pipeline复杂，质量难保证                  | **高** | 分级标注（Level 0→1→2）+ 质量检查指标（Distinct-4 > 0.7, κ > 0.7）+ 先100条小规模验证                                                 |
| **Inference-time Reasoning Collapse**（v2新增）  | **高** | Adaptive Thinking策略 + 三种thinking策略消融实验 + 若Always-Think崩溃则退回Think-to-Internalize                                          |
| **ARNA concurrent work威胁**（v2新增）           | **中** | 明确差异化：SFT-trained vs prompting，跨环境 vs 室内，双系统 vs 单系统。在论文中直接对比                                                 |
| "假Agent"——模型只是记住了模板化的thinking            | **中** | OOD场景的tool call precision分析 + 标注质量消融（实验十）                                                                                |
| Thinking+Tool Call增加输出长度                         | 低           | Adaptive Thinking下60%步骤无thinking。加权平均280ms/step满足2Hz                                                                          |
| 自我纠错数据不足                                       | 中           | 构造偏航轨迹：正确轨迹上随机插入5-10步偏航+纠错段，占比约5%                                                                              |
| SFT后Agent行为退化（灾难性遗忘）                       | 中           | 训练数据中保留30%标准VLN数据（无Agent行为）                                                                                              |
| **回环检测误报（语义相似但位置不同）**（v2新增） | 中           | 双重验证：视觉相似度 > 0.9**且** 拓扑距离 > 10节点。实验九分析误报率                                                               |
| **简单baseline性能出乎意料地好**（v2新增）       | 中           | 若GPS-Threshold Pipeline达到NaviAgent的90%+，转向分析：(a) Agent在哪些场景有显著优势？(b) 是否在GPS不可靠区域/纠错场景有不可替代的价值？ |

---

## 12. 参考文献（v2新增引用标注）

### 核心引用（v2新增）

- Lange et al., 2025. "ARNA: General-Purpose Robotic Navigation via LVLM-Orchestrated Perception, Reasoning, and Acting." arXiv 2506.17462.
- Aux-Think, 2025. "Aux-Think: Exploring Reasoning Strategies for Data-Efficient VLN." arXiv 2505.11886.
- Qi et al., 2025. "VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning." arXiv 2506.17221.
- Nav-R1, 2025. "Nav-R1: Reasoning and Navigation in Embodied Scenes." arXiv 2509.10884.
- ETP-R1, 2025. "ETP-R1: Evolving Topological Planning with Reinforcement Fine-tuning for VLN-CE." arXiv 2512.20940.
- Huang et al., 2026. "TIC-VLA: Think-in-Control VLA for Robot Navigation in Dynamic Environments." arXiv 2602.02459.
- RAGNav, 2026. "RAGNav: A Retrieval-Augmented Topological Reasoning Framework for Multi-Goal VLN." arXiv 2603.03745.
- Castro et al., 2025. "VAMOS: Hierarchical VLA for Capability-Modulated and Steerable Navigation." arXiv 2510.20818.
- TagaVLM, 2026. "TagaVLM: Topology-Aware Global Action Reasoning for VLN." arXiv 2603.02972.

### 已有引用（v1保留）

- DualVLN (2025), InternVLA-N1 (2025), UrbanVLA (2025), CogNav (2025), MapNav (2025), NavFoM (2025), ASCENT (2026), BridgeNav (2026), NavCoT (2024), ReAct (2023).
