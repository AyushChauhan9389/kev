#!/usr/bin/env bash
# Kev on MiniCPM5-2B-Base, run on one local H200 (no Modal): three studies side by side, then pick and score the winner.
#
#   scripts/h200_minicpm5.sh               # setup + train + eval, detached: returns at once, survives closing the terminal / SSH
#   PHASE=train scripts/h200_minicpm5.sh
#   PHASE=eval  scripts/h200_minicpm5.sh   # after training finished (or to re-score)
#   PHASE=status scripts/h200_minicpm5.sh  # is it running, last log lines, GPU memory
#   PHASE=stop  scripts/h200_minicpm5.sh   # stop the run and every lane it started
#   FOREGROUND=1 scripts/h200_minicpm5.sh  # stay attached (Ctrl-C stops it)
#
# Detached runs start in their own session (setsid, stdin closed), so no hangup reaches them; everything goes to
# runs/h200-logs/run.log (lanes: runs/h200-logs/<lane>.log) and the session id to runs/h200-logs/run.pid.
#
# Lanes (experiments/*.json, 2 trials each, the decision-v7 recipe at batch 8 / bf16 / 2 epochs, each trial also scored
# on transfer-v4 development):
#   minicpm5-2b-a       lr 1e-4 with and without --special_embeddings (are MiniCPM5's reserved delimiter rows usable untrained?)
#   minicpm5-2b-b       lr 5e-5, and lr 1e-4 at a second seed
#   qwen35-2b-control   Qwen3.5-2B-Base, the same recipe with no code changes (the base to beat)
# Each lane is its own kev.experiment study on its own --queue, so the three share the GPU. They train with gradient
# checkpointing: three 2B trainings then stay far below 141 GB and together keep the H200 busy. Watch `nvidia-smi`; with
# memory to spare, add lanes (a copy of a plan with other seeds) rather than turning checkpointing off.
# The locked test partition is never read here.
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE=${PHASE:-all}
LANES=${LANES:-"minicpm5-2b-a minicpm5-2b-b qwen35-2b-control"}
SUITE=evals/v7/decision-v7
TRANSFER=evals/v4/transfer-v4
LOGS=runs/h200-logs
BASELINES=${BASELINES:-"jaredpalmer/kev-0.8b jaredpalmer/kev-4b"}
export TOKENIZERS_PARALLELISM=false

running() { [ -f $LOGS/run.pid ] && kill -0 "$(cat $LOGS/run.pid)" 2>/dev/null; }

case $PHASE in
  status)
    if running; then echo "running: session $(cat $LOGS/run.pid)"; else echo "not running"; fi
    [ -f $LOGS/run.log ] && tail -n 15 $LOGS/run.log
    for log in $LOGS/*.log; do [ "$log" = $LOGS/run.log ] || { [ -f "$log" ] && echo "== $log" && tail -n 3 "$log"; }; done
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    exit 0 ;;
  stop)
    if running; then kill -- "-$(cat $LOGS/run.pid)" && echo "stopped session $(cat $LOGS/run.pid) and its lanes"; else echo "not running"; fi
    exit 0 ;;
esac

if [ "${FOREGROUND:-0}" != 1 ] && [ -z "${KEV_H200_DETACHED:-}" ]; then
  if running; then echo "already running (session $(cat $LOGS/run.pid)); PHASE=status or PHASE=stop" >&2; exit 1; fi
  mkdir -p $LOGS
  # a script's background job is not a process-group leader, so setsid execs in place: $! is the new session's leader
  # and its id is the process group PHASE=stop kills (the lanes inherit it)
  KEV_H200_DETACHED=1 setsid nohup "$0" "$@" > $LOGS/run.log 2>&1 < /dev/null &
  echo $! > $LOGS/run.pid
  echo "detached: session $!, PHASE=$PHASE"
  echo "  follow:  tail -f $LOGS/run.log"
  echo "  status:  PHASE=status $0"
  echo "  stop:    PHASE=stop $0"
  exit 0
fi

setup() {
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  uv sync --extra serve
  uv pip install flash-linear-attention   # fast DeltaNet kernels for the Qwen3.5 control lane (as in the Modal image)
  uv run python -m pytest tests/test_unit.py -q
  for lane in $LANES; do   # validates every plan against the suite and fetches its partitions before any GPU time is spent
    uv run python -m kev.experiment --suite $SUITE --plan experiments/$lane.json --out runs/$lane --dry-run > /dev/null
  done
  # ~2 minutes: train MiniCPM5 on the 8-record smoke suite with the lane settings, then check its serving paths agree
  # (packed == rows == prefix cache, delimiter embeddings in the adapter). A problem shows up here, not an hour into a lane.
  rm -rf runs/h200-smoke
  uv run python -m kev.train --suite evals/smoke-v1 --base openbmb/MiniCPM5-2B-Base --base_revision 96a57cd572a02506b4500f54427dca24970c1bac \
    --epochs 2 --batch 8 --accum 1 --dtype bf16 --checkpointing 1 --special_embeddings 1 --lr 1e-4 --device cuda --out runs/h200-smoke
  KEV_MINICPM_SMOKE=runs/h200-smoke uv run --extra serve python -m pytest tests/test_model.py -k minicpm -q -rs
}

train() {
  mkdir -p $LOGS
  local pids=()
  for lane in $LANES; do
    if [ -e runs/$lane ]; then echo "runs/$lane exists; move it away or drop the lane from LANES" >&2; exit 1; fi
    uv run python -m kev.experiment --suite $SUITE --plan experiments/$lane.json --transfer $TRANSFER \
      --out runs/$lane --queue $lane --device cuda > $LOGS/$lane.log 2>&1 &
    pids+=($!); echo "lane $lane: pid $! (log $LOGS/$lane.log)"
    sleep 60   # stagger model loading
  done
  local failed=0
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "lane $(echo $LANES | cut -d' ' -f$((i + 1))) failed; see its log" >&2; failed=1; }
  done
  return $failed
}

evaluate() {
  # the winner: best transfer-v4 development accuracy over every finished trial (model selection on development data only)
  local winner
  winner=$(uv run python - $LANES <<'EOF'
import json, sys
from pathlib import Path
rows = []
for lane in sys.argv[1:]:
    for result in sorted(Path("runs", lane).glob("*/result.json")):
        r = json.loads(result.read_text(encoding="utf-8"))
        cfg = r["provenance"]["config"]
        rows.append((r["transfer"]["clean"]["acc"], r["clean"]["acc"], r["transfer"]["clean"]["brier"], cfg["base"].split("/")[1], cfg["lr"], cfg.get("special_embeddings", 0), cfg["seed"], str(result.parent)))
print(f"{'transfer-v4 acc':>15} {'decision-v7 acc':>15} {'transfer brier':>14}  base / lr / special_emb / seed", file=sys.stderr)
for t, d, b, base, lr, se, seed, path in sorted(rows, reverse=True):
    print(f"{t:15.3f} {d:15.3f} {b:14.3f}  {base} / {lr} / {se} / {seed}   {path}", file=sys.stderr)
print(max(rows)[-1])
EOF
)
  echo "winner: $winner"
  # write the calibrated temperature into the winner's head.pt (fit on its decision-v7 development rows only)
  uv run python scripts/calibrate_checkpoint.py --run $winner/checkpoint --rows $winner/development/rows.json --transfer $winner/transfer/rows.json
  mkdir -p runs/h200-eval
  for suite in evals/v4/transfer-v4 evals/v9/transfer-v9; do
    local name; name=$(basename $suite)
    uv run python -m kev.benchmark --run $winner/checkpoint --suite $suite --device cuda --out runs/h200-eval/winner-$name
    for base in $BASELINES; do
      local tag=${base#*/}
      [ -e runs/h200-eval/$tag-$name ] || uv run python -m kev.benchmark --run $base --suite $suite --device cuda --out runs/h200-eval/$tag-$name
      uv run python -m kev.compare --candidate runs/h200-eval/winner-$name --reference runs/h200-eval/$tag-$name --out runs/h200-eval/winner-vs-$tag-$name
    done
  done
  echo "done: reports in runs/h200-eval/ (report.json per run, compare output per pair); winner checkpoint $winner/checkpoint"
}

case $PHASE in
  all) setup; train; evaluate; echo "all phases finished" ;;
  setup) setup ;;
  train) train ;;
  eval) evaluate ;;
  *) echo "PHASE must be all, setup, train, eval, status or stop" >&2; exit 2 ;;
esac
