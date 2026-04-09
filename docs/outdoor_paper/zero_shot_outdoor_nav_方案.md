# Zero-Shot 室外导航方案：Qwen3.5-9B Prompting + MetaUrban

> 目标：**不微调 VLM**，仅用 Qwen3.5-9B 的 prompting 能力，在 MetaUrban 中实现 point-goal 室外导航。先跑通闭环，验证 VLM 的 zero-shot 导航决策能力。

---

## 方案总览

```
                 MetaUrban 仿真环境
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   RGB 图像         Agent 位置        Goal 位置
   (前视相机)       (世界坐标)        (世界坐标)
        │               │               │
        ▼               ▼               ▼
   ┌─────────────────────────────────────────┐
   │         Qwen3.5-9B  Prompting           │
   │                                         │
   │  输入: RGB 图像 + 场景描述 prompt        │
   │       + GPS 相对方向/距离               │
   │       + 历史动作摘要                    │
   │                                         │
   │  输出: 文本决策                          │
   │       → 解析为 [steering, acceleration]  │
   └─────────────────────────────────────────┘
                        │
                        ▼
               MetaUrban env.step(action)
                        │
                        ▼
                   下一帧观测...
```

**核心思路**：模仿 SocialNav 的 Brain Module，让 VLM 做**场景理解 + 导航决策**，但不训练，纯 prompt。

---

## 环境与传感器配置

### MetaUrban 环境

```python
config = {
    "map": "SXSCSXS",         # 含弯道、十字路口、环岛
    "object_density": 0.3,
    "crswalk_density": 1,
    "horizon": 500,
    "num_scenarios": 20,
    "image_observation": True,
    "stack_size": 1,
    "norm_pixel": False,       # uint8 RGB
    "sensors": {
        "rgb_camera": (RGBCamera, 640, 480),
    },
    "agent_type": "coco",
}
```

### 传感器数据获取

每步从 MetaUrban 获取：

| 数据 | 来源 | 格式 |
|------|------|------|
| **前视 RGB** | `obs["image"]` | 640×480 uint8 |
| **Agent 位置** | `agent.position` | (x, y) 世界坐标 |
| **Agent 朝向** | `agent.heading_theta` | 弧度 |
| **Goal 位置** | `agent.navigation.checkpoints[-1]` | (x, y) 世界坐标 |
| **Route completion** | `info["route_completion"]` | float [0, 1] |

### GPS 模拟

从 agent position 和 goal position 计算"GPS 导航信息"：

```python
def compute_gps_info(agent_pos, agent_heading, goal_pos):
    """模拟 GPS 导航信息：相对方向和距离"""
    dx = goal_pos[0] - agent_pos[0]
    dy = goal_pos[1] - agent_pos[1]
    distance = math.sqrt(dx**2 + dy**2)

    # 目标相对于 agent 朝向的角度
    goal_angle = math.atan2(dy, dx)
    relative_angle = goal_angle - agent_heading
    # 归一化到 [-180, 180]
    relative_angle = math.degrees(relative_angle)
    relative_angle = ((relative_angle + 180) % 360) - 180

    # 方向描述
    if abs(relative_angle) < 20:
        direction = "straight ahead"
    elif relative_angle > 0 and relative_angle < 60:
        direction = "slightly to the right"
    elif relative_angle >= 60 and relative_angle < 120:
        direction = "to the right"
    elif relative_angle >= 120:
        direction = "behind you to the right"
    elif relative_angle < 0 and relative_angle > -60:
        direction = "slightly to the left"
    elif relative_angle <= -60 and relative_angle > -120:
        direction = "to the left"
    else:
        direction = "behind you to the left"

    return {
        "distance_m": round(distance, 1),
        "relative_angle_deg": round(relative_angle, 1),
        "direction": direction,
    }
```

---

## VLM Prompt 设计

### System Prompt

```
You are a navigation robot walking on city sidewalks. You receive:
1. A first-person RGB image of what you see
2. GPS information (distance and direction to goal)
3. Your recent action history

Your job: decide the next action to reach the goal safely.

Output format (EXACTLY this, nothing else):
SCENE: [1 sentence describing what you see]
REASONING: [1 sentence about what to do]
ACTION: [one of: GO_STRAIGHT, TURN_LEFT, TURN_RIGHT, SLOW_DOWN, STOP]
```

### Per-step Prompt

```python
def build_prompt(gps_info, action_history, route_completion):
    return f"""## GPS Navigation
- Goal: {gps_info['distance_m']}m away, {gps_info['direction']}
- Relative angle: {gps_info['relative_angle_deg']}°
- Route progress: {route_completion:.0%}

## Recent actions (last 5)
{', '.join(action_history[-5:]) if action_history else 'None'}

## Instructions
Look at the image and decide your next action.
- GO_STRAIGHT: continue forward (good when goal is ahead)
- TURN_LEFT: turn left ~30° (when goal is to the left or obstacle ahead)
- TURN_RIGHT: turn right ~30° (when goal is to the right or obstacle ahead)
- SLOW_DOWN: reduce speed (near obstacles or pedestrians)
- STOP: you have arrived

Respond in the exact format: SCENE / REASONING / ACTION"""
```

### 动作映射

```python
ACTION_MAP = {
    "GO_STRAIGHT": [0.0, 0.5],    # [steering, acceleration]
    "TURN_LEFT":   [-0.5, 0.3],
    "TURN_RIGHT":  [0.5, 0.3],
    "SLOW_DOWN":   [0.0, 0.1],
    "STOP":        [0.0, 0.0],
}

def parse_vlm_response(response_text):
    """解析 VLM 文本输出为 MetaUrban 动作"""
    for line in response_text.strip().split('\n'):
        if line.startswith('ACTION:'):
            action_name = line.split(':')[1].strip().upper()
            for key in ACTION_MAP:
                if key in action_name:
                    return ACTION_MAP[key], key
    return ACTION_MAP["GO_STRAIGHT"], "GO_STRAIGHT"  # fallback
```

---

## 实现步骤

### Step 1: 安装 VLM 推理环境

在 metaurban conda 环境中安装 Qwen3.5-9B 推理依赖：

```bash
conda activate metaurban
pip install transformers @ git+https://github.com/huggingface/transformers.git@main
pip install accelerate
# 下载模型（或用 API）
# 本地推理需要 ~20GB 显存（BF16）→ 4070Ti 16GB 不够
# 方案 A: 用 4-bit 量化（~6GB）
pip install bitsandbytes
# 方案 B: 用 API（推荐，避免显存问题）
```

**推荐方案**: 用 **vLLM server** 或 **Ollama** 在本地跑量化版本，或者用云端 API。4070Ti 16GB 可以跑 4-bit 量化的 Qwen3.5-9B。

### Step 2: VLM 推理封装

```python
# vlm_client.py
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

class VLMNavigator:
    def __init__(self, model_name="Qwen/Qwen3.5-9B-Instruct", quantize_4bit=True):
        if quantize_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_name, quantization_config=bnb_config, device_map="auto"
            )
        else:
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def decide(self, rgb_image, gps_info, action_history, route_completion):
        """输入 RGB 图像 + GPS 信息，输出导航决策"""
        pil_image = Image.fromarray(rgb_image)
        prompt = build_prompt(gps_info, action_history, route_completion)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ]},
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_image], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=150, temperature=0.1)

        response = self.processor.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        action, action_name = parse_vlm_response(response)
        return action, action_name, response
```

### Step 3: 闭环导航脚本

```python
# zero_shot_nav.py
def run_zero_shot_navigation(env, vlm, seed=0, max_steps=300):
    obs, info = env.reset(seed=seed)
    agent = env.agents["default_agent"]
    goal_pos = agent.navigation.checkpoints[-1]  # 终点坐标

    action_history = []
    trajectory = []

    for step in range(max_steps):
        # 1. 获取传感器数据
        rgb = extract_rgb(obs)
        agent_pos = agent.position
        agent_heading = agent.heading_theta

        # 2. 计算 GPS 信息
        gps_info = compute_gps_info(agent_pos, agent_heading, goal_pos)

        # 3. VLM 推理（~1-3 秒/步，zero-shot）
        action, action_name, vlm_response = vlm.decide(
            rgb, gps_info, action_history, info.get("route_completion", 0)
        )
        action_history.append(action_name)

        # 4. 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        agent = env.agents["default_agent"]

        # 5. 记录
        trajectory.append({
            "step": step,
            "position": list(agent.position),
            "gps_info": gps_info,
            "vlm_response": vlm_response,
            "action": action_name,
            "route_completion": info.get("route_completion", 0),
        })

        # 6. 打印
        if step % 10 == 0:
            print(f"Step {step}: {action_name} | "
                  f"dist={gps_info['distance_m']}m {gps_info['direction']} | "
                  f"route={info.get('route_completion', 0):.0%}")

        if terminated or truncated:
            break

    return trajectory
```

---

## 执行计划

| 步骤 | 任务 | 预计时间 |
|------|------|:---:|
| 1 | 在 metaurban 环境中安装 transformers + bitsandbytes | 10 min |
| 2 | 下载 Qwen3.5-9B-Instruct（4-bit 量化版本）| 30 min（取决于网速） |
| 3 | 编写 VLM 推理封装 + prompt 设计 | 30 min |
| 4 | 编写闭环导航脚本 | 30 min |
| 5 | 在 MetaUrban 中跑通一条 zero-shot 轨迹 | 10-30 min（取决于推理速度） |
| 6 | 保存可视化（轨迹图 + VLM 推理文本 + RGB 帧）| 10 min |

**预期推理速度**: 4-bit Qwen3.5-9B 在 4070Ti 上约 1-3 秒/步 → 300 步约 5-15 分钟/episode。

**预期效果**: zero-shot 不会很好（VLM 对具体坐标的空间推理能力有限），但能验证：
1. VLM 能否理解街景图像（行人、建筑、人行道）
2. VLM 能否根据 GPS 方向做出合理的转弯决策
3. 整个闭环流程（MetaUrban → RGB → VLM → action → MetaUrban）是否跑通

---

## 后续优化路线

| 阶段 | 改进 | 预期效果 |
|------|------|---------|
| **v0 (当前)** | Zero-shot prompting，5 个离散动作 | 跑通闭环，SR ~20-40% |
| **v1** | 加多视角（4-view），加历史帧，优化 prompt | SR ~40-60% |
| **v2** | 换 UrbanVerse 仿真（更真实场景），加 depth DPE | SR ~50-70% |
| **v3** | SFT 微调 Qwen3.5-9B + NaVocab tool tokens | SR ~70-85%（对标 SocialNav） |
| **v4** | Stage 3 RFT (SAFE-GRPO) + 社会合规奖励 | SR ~85%+（对标 SocialNav Full） |
