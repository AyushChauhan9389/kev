#!/usr/bin/env bash
# Kev on MiniCPM5-2B-Base, run on one local H200 (no Modal): three studies side by side, then pick and score the winner.
#
#   scripts/h200_minicpm5.sh               # setup + train + eval, detached: returns at once, survives closing the terminal / SSH
#   PHASE=train scripts/h200_minicpm5.sh
#   PHASE=resume scripts/h200_minicpm5.sh  # setup already done: train the unfinished lanes, then eval
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
# Each lane is its own kev.experiment study on its own --queue. On a full H200 (>= 120 GB) the three run side by side,
# otherwise one at a time; PARALLEL=n overrides. Rerunning PHASE=train skips finished lanes and restarts incomplete ones
# (the partial study is kept as runs/<lane>.incomplete-<time>).
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
# Kubernetes pods default to `options ndots:5` plus several search domains, so every PyPI / Hub / GitHub hostname is first
# tried under each search domain. glibc programs (pip, Python, the Hub client) read RES_OPTIONS from the environment:
# ndots:1 resolves those names directly (no root needed). uv's static musl resolver ignores it; see install().
export RES_OPTIONS=${RES_OPTIONS:-"ndots:1 timeout:2 attempts:3"}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-8} UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-8} UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-120}
# set by install() when the environment came from pip (uv could not download): every later `uv run`, in this or a later
# PHASE, then uses .venv as it is instead of trying to sync it
PIP_MARKER=.venv/.kev-installed-with-pip
if [ -f $PIP_MARKER ]; then export UV_NO_SYNC=1; fi

# run.pid names our detached session only while that process is alive, leads its own session and is this script: a pod
# reuses small PIDs soon after a run exits, and PHASE=stop kills the whole process group it names.
running() {
  [ -f $LOGS/run.pid ] || return 1
  local pid; pid=$(cat $LOGS/run.pid)
  [[ $pid =~ ^[0-9]+$ ]] && [ "$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$pid" ] \
    && ps -o args= -p "$pid" | grep -q "h200_minicpm5.sh"
}

case $PHASE in
  status)
    if running; then echo "running: session $(cat $LOGS/run.pid)"; else echo "not running"; fi
    [ -f $LOGS/run.log ] && tail -n 15 $LOGS/run.log
    for log in $LOGS/*.log; do [ "$log" = $LOGS/run.log ] || { [ -f "$log" ] && echo "== $log" && tail -n 3 "$log"; }; done
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    exit 0 ;;
  stop)
    if running; then pid=$(cat $LOGS/run.pid); kill -- "-$pid" && echo "stopped session $pid and its lanes"; else echo "not running"; fi
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
if [ -n "${KEV_H200_DETACHED:-}" ]; then trap 'rm -f $LOGS/run.pid' EXIT; fi   # a finished or failed run leaves no pid behind

install() {
  # flash-linear-attention + triton>=3.7.1: fast DeltaNet kernels for the Qwen3.5 control lane, installed over the lock
  # exactly as modal_app.py's image does (torch 2.8 pins triton 3.4; the Modal image runs the newer one)
  if uv sync --extra serve && uv pip install flash-linear-attention "triton>=3.7.1"; then rm -f $PIP_MARKER; return; fi
  # The uv binary is static musl. musl ignores RES_OPTIONS and gives up on the first search domain that does not answer
  # NXDOMAIN, so on a pod with ndots:5 it cannot resolve files.pythonhosted.org ("Name has no usable address") while
  # glibc can. Install the exact uv.lock pins with pip instead (the venv's Python is glibc), then run uv without syncing.
  echo "uv could not download; installing the uv.lock pins with pip" >&2
  uv export --frozen --extra serve --no-emit-project --no-hashes -o $LOGS/requirements.txt > /dev/null   # offline: reads uv.lock
  [ -x .venv/bin/python ] || uv venv
  .venv/bin/python -m ensurepip --upgrade > /dev/null
  .venv/bin/python -m pip install -q -r $LOGS/requirements.txt
  .venv/bin/python -m pip install -q flash-linear-attention "triton>=3.7.1"   # pip notes torch's triton pin; expected, as above
  .venv/bin/python -m pip install -q --no-deps -e .
  touch $PIP_MARKER; export UV_NO_SYNC=1
}

setup() {
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  install
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

trials_done() { local f=(runs/$1/*/result.json); if [ -e "${f[0]}" ]; then echo ${#f[@]}; else echo 0; fi; }
trials_planned() { uv run python -c "import json, sys; print(len(json.load(open(sys.argv[1]))))" experiments/$1.json; }

# Lanes that fit side by side. Three 2B trainings need well over 71 GB; on a smaller slice (a MIG instance) a second lane
# runs the first out of memory, which this container reports as an NVML INTERNAL ASSERT rather than "CUDA out of memory"
# because it may not query NVML. auto = 3 on a GPU with >= 120 GB (measured through CUDA, which works where nvidia-smi
# reports [Insufficient Permissions]), else 1.
parallel_lanes() {
  if [ "${PARALLEL:-auto}" != auto ]; then echo "$PARALLEL"; return; fi
  uv run python -c "import torch; print(3 if torch.cuda.mem_get_info()[1] >= 120e9 else 1)"
}

train() {
  mkdir -p $LOGS
  local todo=() lane
  for lane in $LANES; do
    local finished planned; finished=$(trials_done $lane); planned=$(trials_planned $lane)
    if [ "$finished" -ge "$planned" ]; then echo "lane $lane: all $planned trials finished, skipping"; continue; fi
    if [ -e runs/$lane ]; then   # an interrupted or failed study cannot be continued in place; keep it for its logs
      local keep; keep=runs/$lane.incomplete-$(date +%Y%m%d-%H%M%S)
      mv runs/$lane $keep; echo "lane $lane: $finished of $planned trials had finished; moved to $keep, rerunning the lane"
    fi
    todo+=($lane)
  done
  local slots; slots=$(parallel_lanes)
  echo "GPU memory: $(uv run python -c "import torch; f, t = torch.cuda.mem_get_info(); print(f'{t / 1e9:.0f} GB total, {f / 1e9:.0f} GB free')"); running $slots lane(s) at a time"
  local failed=0 i=0
  while [ $i -lt ${#todo[@]} ]; do
    local pids=() names=()
    for lane in "${todo[@]:$i:$slots}"; do
      uv run python -m kev.experiment --suite $SUITE --plan experiments/$lane.json --transfer $TRANSFER \
        --out runs/$lane --queue $lane --device cuda > $LOGS/$lane.log 2>&1 &
      pids+=($!); names+=($lane); echo "lane $lane: pid $! (log $LOGS/$lane.log)"
      if [ $slots -gt 1 ]; then sleep 60; fi   # stagger model loading
    done
    for j in "${!pids[@]}"; do
      if wait "${pids[$j]}"; then echo "lane ${names[$j]} finished"; else echo "lane ${names[$j]} failed; see $LOGS/${names[$j]}.log" >&2; failed=1; fi
    done
    i=$((i + slots))
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
  resume) train; evaluate; echo "all phases finished" ;;   # after setup already ran: unfinished lanes, then the evaluation
  eval) evaluate ;;
  *) echo "PHASE must be all, setup, train, resume, eval, status or stop" >&2; exit 2 ;;
esac
