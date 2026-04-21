"""
汇总 0419 消融实验 (quick_16) 结果并生成对比报告。

扫描 output/eval/val_seen_<tag>_<timestamp>/ 下的 summary.json + ablation.json,
按 tag 聚合 (同 tag 保留最新 timestamp), 打印表格 + 写 markdown 报告。
"""
import os
import sys
import json
import glob
import re
from pathlib import Path

EVAL_DIR = Path("output/eval")

TAG_ORDER = [
    # Phase 1
    "baseline", "pixel", "v1", "v4",
    "sem_img", "sem_none",
    "nomem", "act_only",
    # Phase 2
    "img0", "img4", "img16",
    "pose0", "pose5", "pose10",
]

TAG_DESC = {
    "baseline":  "3v + direction + text + img8/act20/pose20 (当前主干)",
    "pixel":     "输出 pixel goal (v,vx,vy)",
    "v1":        "仅 front 视角",
    "v4":        "4 视角 (含 back)",
    "sem_img":   "语义图以俯视图片输入",
    "sem_none":  "不传语义图",
    "nomem":     "完全无记忆 (img=0 act=0 pose=0 sem=none)",
    "act_only":  "仅决策动作记忆 (img=0 pose=0)",
    "img0":      "图片记忆 = 0",
    "img4":      "图片记忆 = 4",
    "img16":     "图片记忆 = 16",
    "pose0":     "位姿记忆 = 0",
    "pose5":     "位姿记忆 = 5",
    "pose10":    "位姿记忆 = 10",
}


def collect():
    # tag -> (latest_dir, summary, ablation)
    results = {}
    for d in sorted(EVAL_DIR.glob("val_seen_*_*")):
        m = re.match(r"val_seen_(.+?)_(\d{8}_\d{6})$", d.name)
        if not m:
            continue
        tag, ts = m.group(1), m.group(2)
        s = d / "summary.json"
        a = d / "ablation.json"
        if not s.exists() or not a.exists():
            continue
        summary = json.loads(s.read_text())
        ablation = json.loads(a.read_text())
        prev = results.get(tag)
        if prev is None or ts > prev[0]:
            results[tag] = (ts, summary, ablation, d)
    return results


def fmt_row(tag, r):
    if r is None:
        return f"| `{tag}` | {TAG_DESC.get(tag, '-')} | n/a | n/a | n/a | n/a |"
    _ts, s, _a, _d = r
    return (f"| `{tag}` | {TAG_DESC.get(tag, '-')} | "
            f"{s['success_rate']:.1f}% | "
            f"{s['avg_spl']:.3f} | "
            f"{s['avg_distance_to_goal']:.2f} | "
            f"{s['total_time_s']:.0f}s |")


def main():
    res = collect()
    if not res:
        print("没有找到任何 val_seen_*_* 结果目录")
        sys.exit(1)

    baseline = res.get("baseline")
    print(f"已收集 {len(res)} 个 tag: {sorted(res)}\n")

    # markdown
    lines = []
    eval_set = baseline[2].get("eval_set") if baseline else "(unknown)"
    n_ep = baseline[1].get("n_episodes", "?") if baseline else "?"
    lines.append(f"# 消融实验结果 ({eval_set}, System1 only)")
    lines.append("")
    lines.append(f"**数据**: `{eval_set}` 评测集 ({n_ep} episodes, val_seen)  ")
    lines.append("**配置**: 仅快系统 (System1, --no-planner)；每 episode 上限 100 步；"
                 "VLM = Qwen3-VL-8B 通过 vLLM 服务")
    lines.append("")
    if baseline:
        bs = baseline[1]
        lines.append(f"**baseline**: SR={bs['success_rate']:.1f}%  "
                     f"SPL={bs['avg_spl']:.3f}  "
                     f"avg_dist={bs['avg_distance_to_goal']:.2f}m")
        lines.append("")

    # Phase 1
    lines.append("## Phase 1 — 主实验")
    lines.append("")
    lines.append("| Tag | 说明 | SR | SPL | avg dist | 耗时 |")
    lines.append("|-----|------|----|----|---------|------|")
    for tag in TAG_ORDER[:8]:
        lines.append(fmt_row(tag, res.get(tag)))
    lines.append("")

    # Phase 2
    lines.append("## Phase 2 — 历史记忆长度扫描")
    lines.append("")
    lines.append("| Tag | 说明 | SR | SPL | avg dist | 耗时 |")
    lines.append("|-----|------|----|----|---------|------|")
    for tag in TAG_ORDER[8:]:
        lines.append(fmt_row(tag, res.get(tag)))
    lines.append("")

    # 相对 baseline 的 delta
    if baseline:
        bsr = baseline[1]["success_rate"]
        bspl = baseline[1]["avg_spl"]
        lines.append("## 相对 baseline 的 ΔSR / ΔSPL")
        lines.append("")
        lines.append("| Tag | ΔSR (pp) | ΔSPL | 判定 |")
        lines.append("|-----|----------|------|------|")
        for tag in TAG_ORDER:
            if tag == "baseline" or tag not in res:
                continue
            s = res[tag][1]
            dsr = s["success_rate"] - bsr
            dspl = s["avg_spl"] - bspl
            if dsr > 5:
                verdict = "✅ 有益"
            elif dsr < -5:
                verdict = "❌ 有害"
            else:
                verdict = "⚪ 持平 (±5pp 内)"
            lines.append(f"| `{tag}` | {dsr:+.1f} | {dspl:+.3f} | {verdict} |")
        lines.append("")

    # 来源目录
    lines.append("## 原始结果目录")
    lines.append("")
    for tag in TAG_ORDER:
        r = res.get(tag)
        if r is not None:
            lines.append(f"- `{tag}` → `{r[3]}`")
    lines.append("")

    report_path = EVAL_DIR / "ablation_0419_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # stdout 简版
    for line in lines:
        print(line)
    print(f"\n报告已写入: {report_path}")


if __name__ == "__main__":
    main()
