"""
根据 val_seen_20260415_013646 评测结果构建 50-episode 评测集 (medium_50).

目标: baseline SR ≈ 50% — 25 成功 + 25 失败; 场景均匀分布;
覆盖短/中/长三档 geodesic 距离。
"""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

REF_CSV = Path("output/eval/val_seen_20260415_013646/results.csv")
DATASET = Path("data/vln_ce/R2R_VLNCE_v1-3/val_seen/val_seen.json.gz")
OUT_PATH = Path("src/scripts/eval_sets/medium_50.json")

TARGET = 50
N_SUCC = 25
N_FAIL = 25
NEAR_MISS_DIST = 5.0


def load_ref():
    with open(REF_CSV) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        if r.get("error"):
            continue
        try:
            out[int(r["episode_id"])] = {
                "success": float(r["success"]) > 0,
                "dist": float(r["distance_to_goal"]),
                "spl": float(r["spl"]),
            }
        except ValueError:
            pass
    return out


def load_dataset():
    with gzip.open(DATASET, "rt") as f:
        data = json.load(f)
    out = {}
    for ep in data["episodes"]:
        out[ep["episode_id"]] = {
            "scene": ep["scene_id"].replace("mp3d/", "").split("/")[0],
            "geo": float(ep["info"]["geodesic_distance"]),
            "inst": ep["instruction"]["instruction_text"].strip(),
        }
    return out


def pick_scene_balanced(pool, k):
    """pool: list of (ep_id, meta). 先每场景 1 条 → 均衡增量, 直到 k 条。"""
    by_scene = defaultdict(list)
    for ep_id, meta in pool:
        by_scene[meta["scene"]].append((ep_id, meta))
    # 每个场景按 geo 排序以得到稳定的 top-pick
    for s in by_scene:
        by_scene[s].sort(key=lambda x: x[1]["geo"])
    picks = []
    scenes = sorted(by_scene)
    round_idx = 0
    while len(picks) < k and any(by_scene.values()):
        for s in scenes:
            if not by_scene[s]:
                continue
            picks.append(by_scene[s].pop(0))
            if len(picks) >= k:
                break
        round_idx += 1
        if round_idx > 100:
            break
    return picks


def bucket_geo(geo):
    if geo < 6.0:
        return "short"
    if geo < 10.0:
        return "mid"
    return "long"


def main():
    ref = load_ref()
    ds = load_dataset()

    merged = []
    for ep_id, r in ref.items():
        d = ds.get(ep_id)
        if d is None:
            continue
        merged.append((ep_id, {**r, **d}))

    succ_pool = [(e, m) for e, m in merged if m["success"]]
    near_fail = [(e, m) for e, m in merged if not m["success"] and m["dist"] < NEAR_MISS_DIST]
    far_fail = [(e, m) for e, m in merged if not m["success"] and m["dist"] >= NEAR_MISS_DIST]

    print(f"源数据: {len(merged)} ep | 成功 {len(succ_pool)} | 近失败 {len(near_fail)} | 远失败 {len(far_fail)}")

    succ_pick = pick_scene_balanced(succ_pool, N_SUCC)
    # 失败: 先 near-miss, 不够再从 far 补
    fail_pick = pick_scene_balanced(near_fail, min(N_FAIL, len(near_fail)))
    if len(fail_pick) < N_FAIL:
        rest = N_FAIL - len(fail_pick)
        fail_pick += pick_scene_balanced(far_fail, rest)

    picks = succ_pick + fail_pick
    picks.sort(key=lambda x: x[0])

    episode_ids = [ep for ep, _ in picks]
    scene_set = sorted({m["scene"] for _, m in picks})

    # 分布统计
    buckets = defaultdict(lambda: [0, 0])  # bucket -> [succ, fail]
    for _, m in picks:
        buckets[bucket_geo(m["geo"])][0 if m["success"] else 1] += 1
    dist_summary = {b: {"succ": c[0], "fail": c[1]} for b, c in buckets.items()}

    by_scene = defaultdict(lambda: [0, 0])
    for _, m in picks:
        by_scene[m["scene"]][0 if m["success"] else 1] += 1

    eval_set = {
        "name": "medium_50",
        "description": (
            "中规模评测集: 50 episodes, 基于 val_seen_20260415_013646 评测结果筛选, "
            f"baseline SR ≈ 50% ({N_SUCC} 成功 + {N_FAIL} 失败), "
            f"覆盖 {len(scene_set)} 个场景, geodesic 分布 short/mid/long"
        ),
        "split": "val_seen",
        "baseline_sr": 50.0,
        "selection_criteria": (
            f"{N_SUCC} 成功 + {N_FAIL} 失败 (优先 dist<{NEAR_MISS_DIST}m 近失败), "
            "场景均衡采样 (先每场景 1 条, 再按 geo 升序轮询), 源 val_seen_20260415_013646"
        ),
        "stats": {
            "n_scenes": len(scene_set),
            "n_success": N_SUCC,
            "n_fail": N_FAIL,
            "by_geo_bucket": dist_summary,
            "by_scene_succ_fail": {s: {"succ": v[0], "fail": v[1]} for s, v in sorted(by_scene.items())},
        },
        "episode_ids": episode_ids,
        "episodes": [
            {
                "id": ep,
                "scene": m["scene"],
                "geo_dist": round(m["geo"], 1),
                "baseline_success": int(m["success"]),
                "baseline_dist": round(m["dist"], 2),
                "baseline_spl": round(m["spl"], 3),
                "note": m["inst"][:80] + ("..." if len(m["inst"]) > 80 else ""),
            }
            for ep, m in picks
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(eval_set, indent=2, ensure_ascii=False))
    print(f"\n写入: {OUT_PATH}")
    print(f"  总计 {len(episode_ids)} ep, {len(scene_set)} 个场景")
    print(f"  目标 baseline SR = {N_SUCC/TARGET*100:.1f}%")
    print(f"  geo 分布: {dist_summary}")
    print(f"  场景分布 (top):")
    top_scenes = sorted(by_scene.items(), key=lambda x: -(x[1][0]+x[1][1]))[:10]
    for s, c in top_scenes:
        print(f"    {s}: succ={c[0]} fail={c[1]}")


if __name__ == "__main__":
    main()
