"""Reorganize 参考文献简介.md: 13 categories → 6 categories, preserve all papers."""
import re

with open('docs/proposal/参考文献简介.md', 'r') as f:
    content = f.read()

# ── Step 1: Extract overview section (before first ##) ──
overview_end = content.find('\n## 1.')
overview_section = content[:overview_end]

# ── Step 2: Extract all paper blocks (### title → next ### or ## ) ──
# Split by ### but keep the delimiter
parts = re.split(r'(?=^### )', content, flags=re.MULTILINE)
papers = {}
for part in parts:
    if not part.startswith('### '):
        continue
    # Skip subsection headers like "### 12.1 Frontier..."
    first_line = part.split('\n')[0]
    if re.match(r'### \d+\.\d+', first_line):
        continue
    # Use first line as key
    papers[first_line.strip()] = part.rstrip('\n') + '\n'

print(f"Extracted {len(papers)} paper entries")

# ── Step 3: Define new 6-category structure ──
categories = {
    "## 1. 导航基础架构与双系统": {
        "desc": "> 双系统架构（System 2 慢规划 + System 1 快执行）、VLM + DiT 联合训练、异步推理等基础设计。",
        "papers": [
            "### Ground Slow, Move Fast",  # DualVLN
            "### InternVLA-N1",
            "### GR00T N1",
            "### RationalVLA",
            "### TIC-VLA",
            "### HiRT",
            "### DiffusionVLA",
        ]
    },
    "## 2. 室内导航：VLN 指令跟随与推理": {
        "desc": "> 给定 turn-by-turn 导航指令，跟随指令到达目标。含 CoT 推理、强化微调、记忆增强方法。",
        "papers": [
            "### NavCoT",
            "### Aux-Think",
            "### EvolveNav",
            "### Nav-R1",
            "### VLN-R1",
            "### ETP-R1",
            "### JanusVLN",
            "### MapNav",
            "### Mem4Nav",
            "### RAGNav",
            "### TagaVLM",
            "### Uni-NaVid",
            "### NavFoM",
        ]
    },
    "## 3. 室内导航：无图探索式（ObjectNav / Goal-Nav）": {
        "desc": "> 给定目标描述（\"find a chair\"/\"go to kitchen\"），无路线指令，自主探索环境找到目标。按方法类型分为：Frontier+VLM/LLM 评分、VLM 微调、端到端 VLA、层次化系统、认知系统。",
        "papers": [
            "### CogNav",
            "### L3MVN",
            "### VLFM",
            "### SG-Nav",
            "### ApexNav",
            "### WMNav",
            "### ASCENT",
            "### AERR-Nav",
            "### FiLM-Nav",
            "### VLingNav",
            "### SysNav",
            "### PanoNav",
        ]
    },
    "## 4. 室外与跨环境导航": {
        "desc": "> 室外城市 route following、室内外过渡（BridgeNav）、跨环境统一导航。",
        "papers": [
            "### UrbanVLA",
            "### UrbanNav",
            "### CityWalker",
            "### BridgeNav",
            "### VAMOS",
        ]
    },
    "## 5. Agent 框架与工具调用": {
        "desc": "> 通用 LLM/VLM Agent 框架（ReAct）、工具增强导航（ARNA、CoINS）、Tool-as-Token 范式（ToolkenGPT、ToolGen）。",
        "papers": [
            "### ReAct",
            "### ARNA",
            "### CoINS",
            "### ToolkenGPT",
            "### ToolGen",
            "### Re-Initialization Token Learning",
        ]
    },
    "## 6. VLA 架构设计：Tokenization / Reasoning / Memory / Depth": {
        "desc": "> 动作 tokenization、latent reasoning、可微记忆、Depth 注入等架构组件设计，主要在操作任务上验证但为 NaviAgent 提供架构参考。",
        "papers": [
            "### FAST",
            "### VQ-VLA",
            "### OpenVLA",
            "### LaRA-VLA",
            "### ECoT",
            "### Fast ECoT",
            "### CoT-VLA",
            "### MolmoAct",
            "### MemoryVLA",
            "### VLM-squared",  # VLM²
            "### SpatialVLA",
            "### SD-VLM",
        ]
    },
}

# ── Step 4: Match papers to categories ──
def find_paper(prefix, papers_dict):
    """Find paper entry by title prefix."""
    for key, val in papers_dict.items():
        if key.startswith(prefix):
            return key, val
    return None, None

output_parts = []

# Overview (rewrite)
output_parts.append("""# NaviAgent v3 参考文献简介

> 本文档汇总 NaviAgent v3 proposal 中引用的全部 55 篇论文，按 6 大类组织。每篇包含：作者（一作+通讯+单位）、论文全名、输入/输出、功能效果、创新点、与 NaviAgent 的关联。

## 各工作与 NaviAgent 的任务覆盖对比

| 工作 | 室内探索 | 室外导航 | 跨环境 | 多楼层 | Agent推理 | 工具/记忆 | 实时 | HM3D SR |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| VLingNav | 是 | - | - | - | AdaCoT | 语义记忆 | 是 | **79.1%** |
| SysNav | 是 | - | - | - | VLM(房间级) | 外部图 | 部分 | **80.8%** |
| FiLM-Nav | 是 | - | - | - | - | - | 是 | 77.0% |
| Uni-NaVid | 是 | - | - | - | - | - | 是 | 73.7% |
| CogNav | 是 | - | - | - | 零样本LLM | 认知地图 | 否 | 72.5% |
| AERR-Nav | 是 | - | - | 是 | 状态机 | 关键点记忆 | 否 | 72.3% |
| ASCENT | 是 | - | - | 是 | 零样本LLM | 多层BEV | 否 | 65.4% |
| NavFoM | 是 | 是 | - | - | - | - | 是 | 45.2%(OVON) |
| ARNA | 是 | - | - | - | prompting | 通用工具库 | 否 | - |
| BridgeNav | →室内 | 是 | 单向 | - | - | - | 是 | - |
| VAMOS | 是 | 是 | 部分 | - | - | - | 是 | - |
| DualVLN | 是(VLN) | - | - | - | - | - | 是 | - |
| UrbanVLA | - | 是 | - | - | - | - | 是 | - |
| **NaviAgent** | **是** | **是** | **双向** | **是** | **SFT latent** | **可微tool+memory** | **是(2Hz)** | **TBD** |
""")

# Build each category
used_papers = set()
for cat_title, cat_info in categories.items():
    output_parts.append(f"\n---\n\n{cat_title}\n\n{cat_info['desc']}\n")

    for prefix in cat_info['papers']:
        key, val = find_paper(prefix, papers)
        if key:
            output_parts.append(f"\n{val}")
            used_papers.add(key)
        else:
            print(f"  WARNING: No match for prefix '{prefix}'")

# Check for uncategorized papers
uncategorized = set(papers.keys()) - used_papers
if uncategorized:
    print(f"\nUncategorized papers ({len(uncategorized)}):")
    for p in sorted(uncategorized):
        print(f"  {p[:80]}")

print(f"\nCategorized: {len(used_papers)}/{len(papers)}")

# Write output
with open('docs/proposal/参考文献简介.md', 'w') as f:
    f.write('\n'.join(output_parts))

print("Done!")
