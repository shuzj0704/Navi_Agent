"""Batch update author info and full titles in 参考文献简介.md"""
import re

# Mapping: old title → (new title, new author line)
UPDATES = {
    "### DualVLN: Ground Slow, Move Fast": (
        "### Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-and-Language Navigation",
        "- **作者**: Meng Wei, Chenyang Wan, Jiaqi Peng 等 | 通讯: **Xihui Liu**（The University of Hong Kong）"
    ),
    "### InternVLA-N1": (
        "### InternVLA-N1: An Open Dual-System Vision-Language Navigation Foundation Model",
        "- **作者**: Meng Wei, Chenyang Wan, Jiaqi Peng 等 | 通讯: **Xihui Liu**（HKU / Shanghai AI Lab）"
    ),
    "### GR00T N1": (
        "### GR00T N1: An Open Foundation Model for Generalist Humanoid Robots",
        "- **作者**: Johan Bjorck, Fernando Castaneda 等 40+ | 通讯: **Yuke Zhu**（NVIDIA Research / UT Austin）"
    ),
    "### UrbanVLA": (
        "### UrbanVLA: A Vision-Language-Action Model for Urban Micromobility",
        "- **作者**: Anqi Li, Zhiyong Wang, Jiazhao Zhang 等 | 通讯: **He Wang**（Peking University, PKU-EPIC）"
    ),
    "### UrbanNav": (
        "### UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories",
        "- **作者**: Yanghong Mei, Yirong Yang, Longteng Guo 等 | 通讯: **Longteng Guo**（CASIA 中科院自动化所）"
    ),
    "### CityWalker": (
        "### CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos",
        "- **作者**: Xinhao Liu, Jintong Li 等 | 通讯: **Chen Feng**（NYU ai4ce Lab）"
    ),
    "### CogNav": (
        "### CogNav: Cognitive Process Modeling for Object Goal Navigation with LLMs",
        "- **作者**: Yihan Cao, Jiazhao Zhang 等 | 通讯: **Kai Xu**（国防科技大学 NUDT）"
    ),
    "### NavCoT": (
        "### NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning",
        "- **作者**: Bingqian Lin, Yunshuang Nie 等 | 通讯: **Xiaodan Liang**（中山大学）"
    ),
    "### Aux-Think": (
        "### Aux-Think: Exploring Reasoning Strategies for Data-Efficient Vision-Language Navigation",
        "- **作者**: Shuo Wang, Yongcai Wang 等 | 通讯: **Zhaoxin Fan**（中国人民大学 / 快手）"
    ),
    "### EvolveNav": (
        "### EvolveNav: Empowering LLM-Based Vision-Language Navigation via Self-Improving Embodied Reasoning",
        "- **作者**: Bingqian Lin, Yunshuang Nie 等 | 通讯: **Xiaodan Liang**（中山大学 / MBZUAI）"
    ),
    "### Nav-R1": (
        "### Nav-R1: Reasoning and Navigation in Embodied Scenes",
        "- **作者**: Qingxiang Liu, Ting Huang 等 | 通讯: **Hao Tang**（Peking University）"
    ),
    "### VLN-R1": (
        "### VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning",
        "- **作者**: Zhangyang Qi, Zhixiong Zhang 等 | 通讯: **Hengshuang Zhao**（HKU / Shanghai AI Lab）"
    ),
    "### ETP-R1": (
        "### ETP-R1: Evolving Topological Planning with Reinforcement Fine-tuning for VLN-CE",
        "- **作者**: Shuhao Ye, Sitong Mao 等 | 通讯: **Yue Wang**（Zhejiang University）"
    ),
    "### ARNA": (
        "### ARNA: General-Purpose Robotic Navigation via LVLM-Orchestrated Perception, Reasoning, and Acting",
        "- **作者**: Bernard Lange*, Anil Yildiz, Mansur Arief 等 | 通讯: **Bernard Lange**（Stanford / JPL-Caltech）"
    ),
    "### CoINS": (
        "### CoINS: Counterfactual Interactive Navigation via Skill-Aware VLM",
        "- **作者**: Kangjie Zhou, Zhejia Wen 等 | 通讯: **Chang Liu**（Peking University）"
    ),
    "### ReAct": (
        "### ReAct: Synergizing Reasoning and Acting in Language Models",
        "- **作者**: Shunyu Yao, Jeffrey Zhao 等 | 通讯: **Shunyu Yao / Karthik Narasimhan**（Princeton / Google Brain）"
    ),
    "### ToolkenGPT": (
        "### ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings",
        "- **作者**: Shibo Hao, Tianyang Liu 等 | 通讯: **Zhiting Hu**（UC San Diego）"
    ),
    "### ToolGen": (
        "### ToolGen: Unified Tool Retrieval and Calling via Generation",
        "- **作者**: Renxi Wang, Xudong Han 等 | 通讯: **Haonan Li**（MBZUAI / University of Melbourne）"
    ),
    "### Re-Init Token Learning": (
        "### Re-Initialization Token Learning for Tool-Augmented Large Language Models",
        "- **作者**: Chenghao Li, Liu Liu 等 | 通讯: **Liu Liu**（北京航空航天大学）"
    ),
    "### FAST": (
        "### FAST: Efficient Action Tokenization for Vision-Language-Action Models",
        "- **作者**: Karl Pertsch, Kyle Stachowicz 等 | 通讯: **Karl Pertsch**（Physical Intelligence / UC Berkeley）"
    ),
    "### VQ-VLA": (
        "### VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers",
        "- **作者**: Yating Wang, Haoyi Zhu 等 | 通讯: **Tong He**（Shanghai AI Laboratory）"
    ),
    "### OpenVLA": (
        "### OpenVLA: An Open-Source Vision-Language-Action Model",
        "- **作者**: Moo Jin Kim*, Karl Pertsch*（共一）等 | 通讯: **Chelsea Finn**（Stanford University）"
    ),
    "### MemoryVLA": (
        "### MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation",
        "- **作者**: Hao Shi, Bin Xie 等 | 通讯: **Gao Huang**（Tsinghua University）"
    ),
    "### Mem4Nav": (
        "### Mem4Nav: Boosting VLN in Urban Environments with Hierarchical Spatial-Cognition Long-Short Memory",
        "- **作者**: Lixuan He, Haoyu Dong 等 | 通讯: **Yong Li**（Tsinghua University）"
    ),
    "### JanusVLN": (
        "### JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for VLN",
        "- **作者**: Shuang Zeng, Dekang Qi 等 | 通讯: **Shuang Zeng**（Xi'an Jiaotong University）"
    ),
    "### MapNav": (
        "### MapNav: A Novel Memory Representation via Annotated Semantic Maps for VLM-based VLN",
        "- **作者**: Lingfeng Zhang, Xiaoshuai Hao 等 | 通讯: **Renjing Xu**（HKUST(GZ) / Tsinghua）"
    ),
    "### RAGNav": (
        "### RAGNav: A Retrieval-Augmented Topological Reasoning Framework for Multi-Goal VLN",
        "- **作者**: Ling Luo, Qianqian Bai | 通讯: **Ling Luo**（西南财经大学）"
    ),
    "### VLM-squared (VLM^2)": (
        "### VLM²: Vision-Language Memory for Spatial Reasoning",
        "- **作者**: Zuntao Liu, Yi Du 等 | 通讯: **Chen Wang**（SUNY Buffalo, Spatial AI & Robotics Lab）"
    ),
    "### LaRA-VLA": (
        "### LaRA-VLA: Latent Reasoning VLA — Latent Thinking and Prediction for Vision-Language-Action Models",
        "- **作者**: Shuanghao Bai, Jing Lyu 等 | 通讯: **Shanghang Zhang**（Peking University）"
    ),
    "### Fast ECoT": (
        "### Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse",
        "- **作者**: Zhekai Duan, Yuan Zhang 等 | 通讯: **Chris Xiaoxuan Lu**（UCL）"
    ),
    "### ECoT (Embodied Chain-of-Thought)": (
        "### ECoT: Robotic Control via Embodied Chain-of-Thought Reasoning",
        "- **作者**: Michal Zawalski, William Chen 等 | 通讯: **Sergey Levine**（UC Berkeley）"
    ),
    "### CoT-VLA": (
        "### CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models",
        "- **作者**: Qingqing Zhao, Yao Lu 等 | 通讯: **Tsung-Yi Lin**（NVIDIA Research）"
    ),
    "### MolmoAct": (
        "### MolmoAct: Action Reasoning Models that can Reason in Space",
        "- **作者**: Jason Lee, Jiafei Duan 等 | 通讯: **Ranjay Krishna**（Allen AI / University of Washington）"
    ),
    "### RationalVLA": (
        "### RationalVLA: A Rational Vision-Language-Action Model with Dual System",
        "- **作者**: Wenxuan Song, Jiayi Chen 等 | 通讯: **Haoang Li**（HKUST(GZ)）"
    ),
    "### TIC-VLA": (
        "### TIC-VLA: Think-in-Control VLA for Robot Navigation in Dynamic Environments",
        "- **作者**: Zhiyu Huang, Yun Zhang 等 | 通讯: **Jiaqi Ma**（UCLA）"
    ),
    "### HiRT": (
        "### HiRT: Enhancing Robotic Control with Hierarchical Robot Transformers",
        "- **作者**: Jianke Zhang, Yanjiang Guo 等 | 通讯: **Jianyu Chen**（Tsinghua University / Shanghai Qi Zhi）"
    ),
    "### DiffusionVLA": (
        "### DiffusionVLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression",
        "- **作者**: Junjie Wen, Yichen Zhu 等 | 通讯: **Yichen Zhu / Feifei Feng**（Midea Group / 华东师大）"
    ),
    "### BridgeNav: Bridging the Indoor-Outdoor Gap": (
        "### BridgeNav: Bridging the Indoor-Outdoor Gap — Vision-Centric Instruction-Guided Embodied Navigation for the Last Meters",
        "- **作者**: Yuxiang Zhao*, Yirong Yang*（共一）等 | 通讯: **Mu Xu**（AMAP CV Lab, Alibaba Group）"
    ),
    "### ASCENT": (
        "### ASCENT: Stairway to Success — Online Floor-Aware Zero-Shot Object-Goal Navigation via LLM-Driven Coarse-to-Fine Exploration",
        "- **作者**: Zeying Gong, Rong Li 等 | 通讯: **Junwei Liang**（HKUST(GZ)）"
    ),
    "### VAMOS": (
        "### VAMOS: A Hierarchical Vision-Language-Action Model for Capability-Modulated and Steerable Navigation",
        "- **作者**: Mateo Guaman Castro 等 | 通讯: **Abhishek Gupta**（University of Washington）"
    ),
    "### NavFoM": (
        "### NavFoM: Embodied Navigation Foundation Model",
        "- **作者**: Jiazhao Zhang, Anqi Li 等 | 通讯: **He Wang**（Peking University, PKU-EPIC）"
    ),
    "### TagaVLM": (
        "### TagaVLM: Topology-Aware Global Action Reasoning for Vision-Language Navigation",
        "- **作者**: Jiaxing Liu, Zexi Zhang 等 | 通讯: **Xiaoyan Li**（北京工业大学）"
    ),
    "### Uni-NaVid": (
        "### Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks",
        "- **作者**: Jiazhao Zhang, Kunyu Wang 等 | 通讯: **He Wang**（Peking University, PKU-EPIC）"
    ),
    "### SpatialVLA": (
        "### SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model",
        "- **作者**: Delin Qu, Haoming Song 等 | 通讯: **Xuelong Li**（Shanghai AI Lab / 西北工大）"
    ),
    "### SD-VLM": (
        "### SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models",
        "- **作者**: Pingyi Chen, Yujing Lou 等 | 通讯: **Jieping Ye**（Alibaba DAMO Academy）"
    ),
    # Section 12 exploration papers
    "### L3MVN: LLM for Visual Target Navigation": (
        "### L3MVN: Leveraging Large Language Models for Visual Target Navigation",
        "- **作者**: Bangguo Yu, Hamidreza Kasaei, Ming Cao | 通讯: **Bangguo Yu**（University of Groningen）"
    ),
    "### VLFM: Vision-Language Frontier Maps": (
        "### VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation",
        "- **作者**: Naoki Yokoyama, Sehoon Ha 等 | 通讯: **Naoki Yokoyama**（Georgia Tech / Boston Dynamics AI Institute）"
    ),
    "### SG-Nav: Scene Graph Prompting for LLM ObjectNav": (
        "### SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot Object Navigation",
        "- **作者**: Hang Yin, Xiuwei Xu 等 | 通讯: **Jiwen Lu**（Tsinghua University）"
    ),
    "### ApexNav: Adaptive Exploration": (
        "### ApexNav: An Adaptive Exploration Strategy for Zero-Shot Object Navigation with Target-centric Semantic Fusion",
        "- **作者**: Mingjie Zhang, Yuheng Du 等 | 通讯: **Boyu Zhou**（SUSTech 南方科技大学, STAR Lab）"
    ),
    "### WMNav: World Model Navigation": (
        "### WMNav: Integrating Vision-Language Models into World Models for Object Goal Navigation",
        "- **作者**: Dujun Nie, Xianda Guo 等 | 通讯: **Long Chen**（HKUST）"
    ),
    "### AERR-Nav: Exploration-Recovery-Reminiscing": (
        "### AERR-Nav: Adaptive Exploration-Recovery-Reminiscing Strategy for Zero-Shot Object Navigation",
        "- **作者**: Jingzhi Huang, Junkai Huang 等 | 通讯: **Yi Wang**"
    ),
    "### FiLM-Nav: Fine-tuned Language Model for Navigation": (
        "### FiLM-Nav: Efficient and Generalizable Navigation via VLM Fine-tuning",
        "- **作者**: Naoki Yokoyama, Sehoon Ha | 通讯: **Naoki Yokoyama**（Georgia Tech / Boston Dynamics AI Institute）"
    ),
    "### VLingNav: Adaptive Reasoning + Linguistic Memory": (
        "### VLingNav: Embodied Navigation with Adaptive Reasoning and Visual-Assisted Linguistic Memory",
        "- **作者**: Shaoan Wang, Yuanfei Luo 等 | 通讯: **Junzhi Yu**（Peking University / ByteDance Seed）"
    ),
    "### SysNav: Multi-Level Systematic Cooperation": (
        "### SysNav: Multi-Level Systematic Cooperation Enables Real-World, Cross-Embodiment Object Navigation",
        "- **作者**: Haokun Zhu, Zongtai Li 等 | 通讯: **Ji Zhang**（Carnegie Mellon University）"
    ),
    "### PanoNav: Panoramic Scene Parsing": (
        "### PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory",
        "- **作者**: Qunchao Jin, Yilin Wu, Changhao Chen | 通讯: **Changhao Chen**"
    ),
}

# Read file
with open("docs/proposal/参考文献简介.md", "r") as f:
    lines = f.readlines()

new_lines = []
i = 0
updated = 0
while i < len(lines):
    line = lines[i].rstrip("\n")
    matched = False
    for old_title, (new_title, new_author) in UPDATES.items():
        if line.strip() == old_title:
            new_lines.append(new_title + "\n")
            i += 1
            # Skip empty line after title
            if i < len(lines) and lines[i].strip() == "":
                new_lines.append("\n")
                i += 1
            # Replace author line
            if i < len(lines) and lines[i].strip().startswith("- **作者**"):
                new_lines.append(new_author + "\n")
                i += 1
            else:
                # No author line found, insert new one
                new_lines.append(new_author + "\n")
            matched = True
            updated += 1
            break
    if not matched:
        new_lines.append(lines[i])
        i += 1

with open("docs/proposal/参考文献简介.md", "w") as f:
    f.writelines(new_lines)

print(f"Updated {updated} entries out of {len(UPDATES)} expected.")
