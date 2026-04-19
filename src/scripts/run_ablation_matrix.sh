#!/bin/bash
# 0419 消融实验矩阵 (quick_16 / 仅 System1 / 100 steps).
# 每次 run 的结果目录: output/eval/val_seen_<tag>_<timestamp>/
# 单次 ~15 min, 合计 14 次 ~3.5 h.
#
# 用法:
#   bash src/scripts/run_ablation_matrix.sh                 # 跑全部
#   bash src/scripts/run_ablation_matrix.sh phase1          # 仅 Phase 1
#   bash src/scripts/run_ablation_matrix.sh phase2          # 仅 Phase 2
#   bash src/scripts/run_ablation_matrix.sh <tag1> <tag2>   # 跑指定 tag

set -u
cd "$(dirname "$0")/../.."

EVAL="conda run -n naviagent --no-capture-output python src/scripts/batch_eval.py \
  --vlm-config src/vlm_server/configs/nav_vlm.yaml \
  --eval-set quick_16 --steps 100"

LOG_DIR="output/eval/ablation_0419_logs"
mkdir -p "$LOG_DIR"

# tag -> extra flags
declare -A CFG=(
  # Phase 1
  [baseline]=""
  [pixel]="--output-mode pixel"
  [v1]="--views front"
  [v4]="--views front,left,right,back"
  [sem_img]="--semantic-mode image"
  [sem_none]="--semantic-mode none"
  [nomem]="--image-memory-len 0 --action-history-len 0 --pose-history-len 0 --semantic-mode none"
  [act_only]="--image-memory-len 0 --pose-history-len 0"
  # Phase 2 (memory length sweep, anchored on baseline)
  [img0]="--image-memory-len 0"
  [img4]="--image-memory-len 4"
  [img16]="--image-memory-len 16"
  [pose0]="--pose-history-len 0"
  [pose5]="--pose-history-len 5"
  [pose10]="--pose-history-len 10"
)

PHASE1=(baseline pixel v1 v4 sem_img sem_none nomem act_only)
PHASE2=(img0 img4 img16 pose0 pose5 pose10)

case "${1:-all}" in
  all)     TAGS=("${PHASE1[@]}" "${PHASE2[@]}") ;;
  phase1)  TAGS=("${PHASE1[@]}") ;;
  phase2)  TAGS=("${PHASE2[@]}") ;;
  *)       TAGS=("$@") ;;
esac

echo "=== Ablation matrix: ${#TAGS[@]} runs ==="
printf '  %s\n' "${TAGS[@]}"
echo

START=$(date +%s)
for tag in "${TAGS[@]}"; do
  if [[ -z "${CFG[$tag]+x}" ]]; then
    echo "[SKIP] unknown tag: $tag"
    continue
  fi
  extra="${CFG[$tag]}"
  log="$LOG_DIR/${tag}.log"
  echo ">>> [$(date '+%H:%M:%S')] run: $tag   extra='$extra'"
  t0=$(date +%s)
  $EVAL --ablation-tag "$tag" $extra >"$log" 2>&1
  rc=$?
  t1=$(date +%s)
  dur=$((t1 - t0))
  # pull metric
  sumpath=$(ls -td output/eval/val_seen_${tag}_* 2>/dev/null | head -1)/summary.json
  if [[ -f "$sumpath" ]]; then
    sr=$(python3 -c "import json; d=json.load(open('$sumpath')); print(f\"SR={d['success_rate']:.1f}% SPL={d['avg_spl']:.3f} dist={d['avg_distance_to_goal']:.2f}\")")
    echo "    done rc=$rc dur=${dur}s  $sr"
  else
    echo "    done rc=$rc dur=${dur}s  (no summary found)"
  fi
done

END=$(date +%s)
echo
echo "=== all done: $((END - START))s total ==="
