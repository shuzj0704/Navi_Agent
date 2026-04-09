# NaviAgent: A Tool-Augmented Cognitive Navigation Agent for Seamless Indoor-Outdoor Long-Range Exploration

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

**BridgeNav**：
- 仅处理"室外→室内"单向过渡，不涉及反向和完整链路

### 1.3 我们的核心思想：把VLM训练成一个会用工具的导航Agent

NaviAgent不是一个"输入→输出"的模型，而是一个**工具增强的认知导航Agent**：

- **它会思考**：每一步先推理当前状态、设定短目标，再决定行动
- **它会建图**：实时构建语义拓扑记忆，记录走过的地方和看到的东西
- **它会查图**：主动查询记忆来做决策（"我走过这里吗？"、"出口在哪个方向？"）
- **它会用导航**：到了室外主动调用路线API，获取导航指引
- **它会自我纠错**：走了一段时间没有进展→查图→发现绕圈→换方向

这不是一个固定pipeline，而是一个**动态的Agent循环**：

```
Observe → Think → [Call Tool?] → Act → Observe → Think → ...
```

---

## 2. Problem Definition: Cross-Environment Long-Range Navigation (CELN)

### 2.1 任务定义

给定一个自然语言任务描述（如"去星巴克买一杯咖啡"），机器人从当前室内位置出发，需要自主到达目的地。

机器人配备：
- 多视角RGB相机（始终可用）
- GPS模块（室外可用，室内不可用/不可信）
- 导航工具API（可获取步行路线）

### 2.2 与现有任务的区别

| 任务 | 环境 | 距离 | 导航工具 | 多楼层 | Agent推理 | 代表Benchmark |
|------|------|------|---------|--------|----------|-------------|
| VLN-CE | 室内单层 | 10-20m | 无 | 否 | 无 | R2R-CE |
| ObjectNav | 室内 | 5-20m | 无 | 部分 | 无/简单 | HM3D |
| Urban Nav | 室外 | 200-2000m | GPS路线 | N/A | 无 | MetaUrban |
| Out-to-In Nav | 室外→室内 | 短距离 | 无 | 否 | 无 | BridgeNav |
| **CELN (ours)** | **室内+室外** | **200-2000m** | **按需调用** | **是** | **Agent推理** | **NaviAgent-Bench** |

---

## 3. Contribution

1. **NaviAgent：首个工具增强的认知导航Agent框架**。将VLM从"条件反射式"的输入→输出模型，转变为具备工具调用能力的认知Agent。Agent能够：实时建图、主动查图、按需调用导航API、自我监控与纠错。关键技术是将Agent的推理-工具调用-行动循环编码为SFT训练数据（ReAct-style CoT），实现端到端可训练（2Hz）且具备Agent行为的导航模型。

2. **CELN任务定义与NaviAgent-Bench评估协议**。定义首个跨环境长程导航任务，构建标准化评估体系（逻辑拼接评估 + GPS渐变过渡区评估 + 真机定量评估），设计CELN专用指标（Transition SR、Mode Stability、Reasoning Accuracy等）。

3. **轻量级语义拓扑记忆（作为Agent的可调用工具）**。融合MapNav的语义地图思想和CogNav的认知地图思想，但以极轻量方式实现。记忆模块作为Agent的**工具之一**被调用——Agent决定何时建图、何时查图，而非固定频率更新。

4. **开源Baseline + Benchmark + 认知推理标注数据**。

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
│  │   8B      │    │  🗺️ Semantic Memory  │       │
│  │           │    │  📍 Route Planner    │       │
│  │ Reasoning │    │  🔍 Progress Checker │       │
│  │ + Action  │    │  🏢 Floor Detector   │       │
│  └─────┬─────┘    └──────────────────────┘       │
│        │                                          │
│        ▼                                          │
│  ┌───────────┐                                    │
│  │ System 1  │    30Hz 避障轨迹                   │
│  │ DiT Policy│──────────────────► 底层控制器      │
│  └───────────┘                                    │
└─────────────────────────────────────────────────┘
```

**System 2（Qwen3-VL-8B，2Hz）**：认知Agent核心——推理、决策、工具调用、pixel goal预测

**Agent Tools**：Agent可以按需调用的工具集
- Semantic Memory（语义拓扑记忆）：建图/查图/回环检测
- Route Planner（路线规划）：调用导航API获取路线
- Progress Checker（进度检查）：判断是否在前进/是否绕圈
- Floor Detector（楼层检测）：判断当前楼层

**System 1（DiT扩散策略，30Hz）**：低层执行——接收latent goal，生成避障轨迹

### 4.2 Agent的工具定义

NaviAgent的VLM被训练为一个具备工具调用能力的Agent。每个工具有明确的输入/输出定义：

| 工具 | 触发条件（Agent自主决定） | 输入 | 输出（注入prompt） |
|------|------------------------|------|------------------|
| `update_map()` | 每次推理自动调用 | 当前位置+视觉特征+语义标签 | — (写入记忆) |
| `query_map()` | Agent主动决定查图 | 查询类型（轨迹/回环/进度） | 文本（轨迹摘要/回环警告/进度报告） |
| `call_route()` | GPS可用时Agent决定调用 | 当前GPS位置+目的地 | 路线waypoints+转弯指令 |
| `check_progress()` | Agent感觉不对劲时调用 | 最近N步的位置序列 | 前进距离/绕圈检测/效率评分 |
| `detect_floor()` | 检测到楼梯/电梯时 | 当前视觉观测 | 当前楼层估计 |

**关键设计：工具调用是Agent自主决定的，不是规则触发的。**

- 固定pipeline方式：GPS < 阈值 → 自动切换到route following（硬编码）
- **Agent方式**：Agent推理"GPS信号变好了，我可以看看导航路线了" → 调用`call_route()` → 根据返回的路线决定是否跟随

这意味着Agent可以学到更灵活的策略：比如GPS信号一般但视觉上能看到目标建筑，Agent可能决定"不需要调导航，我直接朝目标走"。

### 4.3 Agent的推理-行动循环

每一步（2Hz）的完整流程：

```
输入：
  [Observation] 四视角RGB
  [Task] "去星巴克买一杯咖啡"
  [Memory Context] (上次query_map的结果，如果有的话)
  [Route Context] (上次call_route的结果，如果有的话)

Agent输出：
  [Thinking]                          ← 认知推理（SFT训练学会的）
  I'm in a corridor on floor 2. I see an elevator sign ahead on the left.
  My short-term goal is to reach the elevator and go down to floor 1.
  The exit should be on the south side based on the route start point.
  I've been making good progress - no need to check the map right now.

  [Tool Call] (可选)                   ← 工具调用决策
  update_map(scene_type="corridor")   ← 自动建图

  [Action]                             ← 动作输出
  Pixel Goal: (front, 156, 312)
```

**另一个场景——Agent决定查图纠错：**

```
  [Thinking]
  I've been walking for a while but this corridor looks familiar.
  I might be going in circles. Let me check my map.

  [Tool Call]
  query_map(type="loop_check")
  → Response: "WARNING: You visited this area 2 minutes ago.
     Trajectory: office → corridor → turn_right → corridor(HERE, REVISITED).
     Suggest: try going left instead of right at the last junction."

  [Thinking]
  The map confirms I'm going in circles. I should turn around and go left
  at the last junction I passed.

  [Action]
  Pixel Goal: (back, 320, 240)  ← 掉头
```

**另一个场景——Agent决定调用导航：**

```
  [Thinking]
  I just exited the building. I can see open sky. GPS signal should be
  available now. Let me check the navigation route.

  [Tool Call]
  call_route(destination="Starbucks, Building B, 3F")
  → Response: "Route available. GPS confidence: high (HDOP=4.2).
     Next: go straight 120m, then turn left onto Main Street.
     Waypoints: [(5.2, 0.3), (10.1, 0.5), ...]"

  [Thinking]
  Good, I have navigation route now. The next waypoint is straight ahead.
  Switching to route following mode.

  [Action]
  Pixel Goal: (front, 320, 180)
```

### 4.4 认知状态（Agent自主判断，不靠硬编码规则）

Agent通过推理自主判断当前状态，而非由GPS阈值等规则触发：

| 状态 | Agent的典型推理 |
|------|---------------|
| Indoor Exploration | "I'm inside a building, no GPS. Looking for stairs or exit." |
| Floor Transition | "I found the elevator. Going down to floor 1." |
| Exit Seeking | "I'm on floor 1 now. The route starts from the south, so I'm looking for the south exit." |
| Outdoor Route Following | "I'm outside, GPS is good. Following the navigation route." |
| Building Entry | "I can see the target building. Looking for an entrance." |
| Target Seeking | "I'm inside the target building. Looking for Starbucks on floor 3." |
| Self-Correction | "I seem to be going in circles. Let me check my map and change direction." |

**与CogNav的关键区别**：
- CogNav：每步调用LLM做zero-shot状态判断（1-2秒/步，不可实时部署）
- NaviAgent：状态判断嵌入VLM的SFT训练数据中，VLM学会自主推理（2Hz，可实时部署）
- CogNav：5个室内ObjectNav状态
- NaviAgent：7个跨环境状态 + 自我纠错状态

### 4.5 语义拓扑记忆（Agent的核心工具）

**设计原则**：作为Agent的工具被调用，而非固定频率更新。

**数据结构**：增量式语义拓扑图
- 每2-5米创建一个节点
- 节点存储：位置坐标、256维VLM视觉特征（零额外计算）、语义标签、楼层标签
- 边存储：节点间的连接关系和距离

**工具接口**：

`update_map(scene_type)`：Agent每步自动调用，写入新节点
- 输入：当前位置、VLM视觉特征、场景类型
- 操作：创建新节点（如距离上一节点>阈值）

`query_map(type)`：Agent主动决定何时调用
- `type="trajectory"`→ 返回轨迹语义序列（"office→corridor→elevator(↓)→lobby"）
- `type="loop_check"`→ 做回环检测（余弦相似度+语义标签双重验证），返回是否绕圈及建议
- `type="progress"`→ 计算已行进距离、估计剩余距离、效率评分
- `type="floor_info"`→ 返回当前楼层和楼层历史

**轻量性**：500米路线约100-250个节点，<1MB内存，回环检测<1ms。

**与MapNav/CogNav的对比**：

| 维度 | CogNav认知地图 | MapNav ASM | NaviAgent语义拓扑记忆 |
|------|-------------|-----------|-------------------|
| 组成 | Scene Graph+Landmark Graph+Occupancy Map | 多通道2D语义地图+文本标注 | 单一拓扑图 |
| 输入需求 | RGB-D+open-vocab segmentation | RGB-D+语义分割 | 仅RGB（VLM副产物） |
| 调用方式 | 固定频率 | 每步更新 | **Agent按需调用** |
| 计算开销 | 重（3D点云、DBSCAN） | 中（点云投影、标注生成） | 极轻（<1ms） |
| 支持范围 | 室内单层 | 室内单层 | 室内外跨楼层500m+ |

### 4.6 Route Planner工具

Agent到了室外决定调用导航时：

`call_route(destination)`：
- 调用高德步行路线API
- 截取前方40m片段，重采样为20个egocentric航点
- 返回：waypoints + 转弯指令 + GPS置信度标签

**GPS置信度**：HDOP映射为离散标签（high/medium/low/unreliable）

**关键区别**：模式切换不再由GPS阈值硬编码触发，而是Agent通过推理决定是否调用/是否信任route信息。Agent可以学到：
- GPS信号一般但能看到目标→不调route，直接走
- GPS信号好但route指向一堵墙→忽略route，靠视觉
- GPS信号从bad变good→先试探性调route，确认合理后再跟随

### 4.7 VLM选型：Qwen3-VL-8B

选择理由：
- 256K context：四相机+历史+route+memory+reasoning全部塞得下
- 3D Grounding：pixel goal空间推理
- DeepStack多层特征融合：视觉理解质量

训练方案基于InternVLA-N1迁移：LR=1e-5，Vision LR=2e-6，BF16，8×A100。

### 4.8 System 1：DiT扩散策略

复用DualVLN设计，接收System 2的latent goal + 30Hz实时RGB生成避障轨迹。与Agent的工具调用无关。

### 4.9 强化微调

**Sim-RFT**：Plan A用IQL（沿用UrbanVLA验证方案），Plan B用GRPO（沿用VLN-R1方案）。
**Real-RFT**：IQL，校园3-5条路线，8-10小时遥操作。

---

## 5. Training Data

### 5.1 SFT数据构成

| 数据类型 | 来源 | 规模 | 训练能力 |
|---------|------|------|---------|
| 室内VLN | R2R-CE + ScaleVLN(2万子集) | ~30K episodes | Indoor Exploration |
| 室内找出口 | HM3D出口标注（自构造） | ~4K episodes | Exit Seeking |
| 室外Route | MetaUrban PPO Expert | ~5-10K episodes | Route Following |
| 过渡区数据 | Habitat内GPS渐变模拟（自构造） | ~1-2K episodes | Mode Switching |
| GPS噪声增强 | 上述所有×4 | ×4扩增 | GPS Robustness |

### 5.2 Agent行为标注（核心数据创新）

**这是NaviAgent区别于所有现有工作的关键训练数据。**

对每条轨迹的每一步，标注完整的Agent行为：Thinking + Tool Call + Action。

**自动标注pipeline**：

1. **状态标注**：基于规则（GPS值→室内/室外、高度变化→楼层、距出口距离→Exit Seeking等）

2. **Thinking生成**：
   - 基础版：规则模板（"I'm in {scene_type} on floor {floor}. My goal is {goal}."）
   - 增强版：用Qwen3-VL-8B-Instruct对当前帧做captioning，生成更自然的描述

3. **Tool Call标注**：
   - `update_map()`：每步都有
   - `query_map()`：
     - 走了20步以上未到达子目标→标注query_map(type="progress")
     - 当前视觉特征与历史节点相似→标注query_map(type="loop_check")
     - 到达新楼层→标注query_map(type="floor_info")
   - `call_route()`：GPS从unreliable变为medium/high时→标注call_route()
   - `check_progress()`：每30步自动标注一次

4. **Action标注**：pixel goal坐标（沿用InternVLA-N1的标注方式）

**示例训练样本**：

```json
{
  "observation": "<front_img> <left_img> <right_img> <back_img>",
  "task": "Go to Starbucks in Building B",
  "memory_context": "Trajectory: office→corridor\nFloor: 2\nProgress: 15m/500m",
  "thinking": "I'm in a corridor on floor 2. I can see an exit sign pointing left. My short-term goal is to follow the exit sign to find stairs or elevator down to floor 1.",
  "tool_calls": ["update_map(scene_type='corridor')"],
  "action": "pixel_goal: (left, 234, 187)"
}
```

**过渡区样本（关键）**：

```json
{
  "observation": "<front_img showing glass door and outdoor view>",
  "task": "Go to Starbucks in Building B",
  "memory_context": "Trajectory: office→corridor→elevator(↓)→lobby\nFloor: 1\nProgress: 45m/500m",
  "route_context": "unavailable",
  "thinking": "I can see the glass entrance door ahead with sunlight outside. I'm about to exit the building. Once outside, I should try to get navigation route. The route to Starbucks should start from the south side of this building.",
  "tool_calls": ["update_map(scene_type='lobby_exit')", "call_route(destination='Starbucks, Building B')"],
  "action": "pixel_goal: (front, 320, 250)"
}
```

**自我纠错样本**：

```json
{
  "observation": "<front_img showing a familiar-looking corridor>",
  "task": "Go to Starbucks in Building B",
  "memory_context": "Trajectory: ...→corridor→right_turn→corridor\nProgress: stagnant for 8 steps",
  "thinking": "This corridor looks familiar. I've been walking for a while without progress. Let me check if I'm going in circles.",
  "tool_calls": ["query_map(type='loop_check')"],
  "tool_response": "LOOP DETECTED: You visited this area 45 seconds ago. Last junction was 3 steps back. Suggest: turn left instead of right.",
  "thinking_2": "I am indeed going in circles. I should go back and try the left path at the last junction.",
  "action": "pixel_goal: (back, 320, 240)"
}
```

### 5.3 过渡区数据构造

在Habitat长轨迹上模拟GPS过渡：
- **类型A（出建筑）**：GPS=unreliable → 渐变 → GPS=high + route出现
- **类型B（进建筑）**：反向
- **类型C（GPS波动）**：medium/low交替
- **类型D（突变）**：GPS直接跳变

过渡段的Agent行为标注重点：Agent何时决定调用`call_route()`、何时决定停止信任route。

---

## 6. Evaluation

### 6.1 NaviAgent-Bench 评估体系

**第一层：逻辑拼接评估（E2E能力）**
- Indoor段（Habitat+HM3D）+ Outdoor段（MetaUrban），1000-2000 episodes

**第二层：GPS渐变过渡区评估（切换能力）**
- 50个HM3D半室外场景，200 episodes

**第三层：GPS噪声鲁棒性（鲁棒性）**
- 固定100 episodes，σ={2,5,10,20,50}m

### 6.2 Baseline对比

| 方法 | 类型 | Agent推理 | 工具调用 |
|------|------|----------|---------|
| DualVLN Only | 纯VLN | 无 | 无 |
| UrbanVLA Only | 纯Route | 无 | 固定GPS |
| CogNav (adapted) | 认知+zero-shot LLM | 有(LLM) | 固定频率 |
| Oracle Switch | 组合+Oracle切换 | 无 | Oracle |
| **NaviAgent** | 认知+端到端Agent | **有(SFT训练)** | **Agent按需** |
| NaviAgent w/o Thinking | 消融 | 无 | 按需 |
| NaviAgent w/o Tools | 消融 | 有 | 无 |
| NaviAgent w/o Memory | 消融 | 有 | 无记忆 |

### 6.3 实验列表

| 实验 | 目的 |
|------|------|
| 实验一：室内VLN | 验证不退化（R2R-CE, RxR-CE） |
| 实验二：室外Route Following | 验证Route能力（MetaUrban） |
| 实验三：CELN主实验 | 核心E2E评估 |
| 实验四：GPS噪声鲁棒性 | SR vs σ曲线 |
| 实验五：RFT消融 | IQL/GRPO有效性 |
| 实验六：Agent行为消融 | Thinking/Tool/Memory各自贡献 |
| 实验七：工具调用分析 | Agent何时调用哪个工具？调用频率？准确性？ |
| 实验八：自我纠错分析 | Agent检测到多少次绕圈？纠错成功率？ |
| 实验九：真机定量评估 | 校园10-15条路线 |

### 6.4 关键指标

标准指标：SR、SPL、NE、nDTW

CELN专用指标：
- E2E SR / E2E SPL
- Transition SR / Transition Reaction Time / Mode Stability
- GPS Noise Robustness曲线 / Route Reliance曲线

Agent专用指标：
- **Reasoning Accuracy**：认知状态判断准确率
- **Tool Call Precision**：工具调用的合理性（不该调的时候调了=false positive）
- **Self-Correction Rate**：自我纠错的成功率
- **Loop Detection Rate**：绕圈检测率和误报率

---

## 7. Related Work

### 7.1 Vision-Language Navigation
DualVLN (2025)、InternVLA-N1 (2025)、JanusVLN (2025)、StreamVLN (2025)——室内VLN SOTA，但是被动的输入→输出模式，无Agent推理。

### 7.2 室外导航与Route Following
UrbanVLA (2025)、UrbanNav (AAAI 2026)、CityWalker (CVPR 2025)——室外导航，不处理室内和过渡。

### 7.3 认知推理导航
CogNav (Cao et al., 2025)——最相关的工作。LLM驱动的认知状态机+异构认知地图。ObjectNav SOTA（HM3D 72.5%）。但zero-shot、仅室内单层、计算重。**NaviAgent的核心区别是将Agent行为从zero-shot LLM推理转化为端到端SFT训练，且扩展到跨环境。**

### 7.4 Chain-of-Thought导航
NavCoT (AAAI 2024)——VLN的CoT推理。但不涉及工具调用，仅做推理链式输出。

### 7.5 语义地图与空间记忆
MapNav (Zhang et al., 2025)——ASM替代历史帧。VoroNav——Voronoi图+LLM接口。INHerit-SG——层次化场景图。**NaviAgent的记忆是Agent的工具，按需调用而非固定更新。**

### 7.6 多楼层导航
ASCENT (Gong et al., 2026)——多楼层零样本ObjectNav。

### 7.7 室内外过渡
BridgeNav (Zhao et al., 2026)——仅out-to-in单向过渡。

### 7.8 导航基础模型与Agent
NavFoM (Zhang et al., 2025)——跨任务跨平台。ReAct (Yao et al., 2023)——推理+行动的通用Agent范式，NaviAgent将其应用于具身导航。

---

## 8. Novelty分析

### 8.1 NaviAgent的核心创新

**不是DualVLN + UrbanVLA的拼凑，而是一个范式转变：从"模型"到"Agent"。**

| 维度 | 现有方法（模型范式） | NaviAgent（Agent范式） |
|------|-----------------|-------------------|
| 决策方式 | 输入→输出（条件反射） | 思考→工具调用→行动（认知过程） |
| 模式切换 | GPS阈值硬编码 | Agent推理决策（灵活） |
| 空间记忆 | 固定频率更新/不更新 | Agent按需查询（主动） |
| 错误恢复 | 不能 | 自我监控+纠错 |
| 训练方式 | 标准SFT（观察→动作） | ReAct-style SFT（思考→工具→动作） |

### 8.2 审稿人可能的质疑与应对

| 质疑 | 应对 |
|------|------|
| "Agent只是在SFT数据中加了Thinking和Tool Call文本，本质还是SFT" | 训练方式确实是SFT，但数据格式的改变带来了行为模式的改变——模型学会了何时调用工具、何时纠错。实验六七八的消融和分析证明这些行为是真实学到的而非虚假的。 |
| "工具调用是否增加了推理延迟" | 工具本身极轻量（map操作<1ms，route API已预缓存），主要开销是VLM多输出几十个thinking token。256K context下2Hz推理可维持。 |
| "自我纠错在训练数据中如何标注？模型没走过错路怎么学纠错？" | 在训练数据中故意引入少量"走错→查图→发现→纠正"的轨迹（基于正确轨迹人工添加偏航段）。 |
| "逻辑拼接benchmark的限制" | 诚实承认；GPS渐变评估+真机实验弥补。 |

---

## 9. Timeline（16周）

| 阶段 | 周 | 任务 |
|------|---|------|
| Phase 1 数据 | 1-4 | 环境搭建、数据下载/渲染、出口标注、MetaUrban收集、**Agent行为标注pipeline开发**、过渡区数据 |
| Phase 2 SFT | 5-7 | Qwen3-VL-8B SFT（含Agent行为数据）、Benchmark构建、实验一二 |
| Phase 3 RFT+实验 | 8-10 | Sim-RFT(IQL/GRPO)、实验三~八 |
| Phase 4 真机 | 11-13 | 遥操作+Real-RFT、真机10-15条路线 |
| Phase 5 写作 | 14-16 | 论文+开源准备 |

---

## 10. 投稿策略

**论文定位**：Method + Benchmark双贡献
- Method：首个工具增强认知导航Agent（从"模型"到"Agent"的范式转变）
- Benchmark：CELN任务+NaviAgent-Bench

**推荐venue**：
- CoRL 2026（Robotics+Agent，非常合适）
- NeurIPS 2026（Agent+Embodied AI track）
- RSS 2026
- RA-L

---

## 11. 风险与应对

| 风险 | 应对 |
|------|------|
| Agent行为标注pipeline复杂，质量难保证 | 先100条轨迹小规模验证，确认后再大规模；分层标注（规则基础版→VLM增强版→人工检查） |
| Thinking+Tool Call增加输出长度 | 约50-100 token/步，Qwen3-VL 256K context充裕；可做thinking长度消融 |
| "假Agent"——模型只是记住了模板化的thinking | 设计多样化的场景和thinking表述；实验七八分析工具调用的适当性 |
| 自我纠错数据不足 | 构造偏航轨迹：在正确轨迹上随机插入5-10步偏航，然后接纠错段 |
| SFT后Agent行为退化（灾难性遗忘） | 训练数据中保留30%的标准VLN数据（无Agent行为），防止遗忘基础导航能力 |
