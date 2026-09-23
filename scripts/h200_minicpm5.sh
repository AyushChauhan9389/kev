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
# Every trial runs as its own job on a GPU slot (see gpu_slots and train): all visible GPUs by default (GPUS=0,1 picks),
# 1-3 runs per GPU by memory (SLOTS=n overrides). 4 trials on 4 GPUs run at once; on a 32 GB MIG slice, one at a time.
# The locked test partition is never read here.
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE=${PHASE:-all}
LANES=${LANES:-"minicpm5-2b-a minicpm5-2b-b"}   # add qwen35-2b-control for the Qwen3.5-2B baseline
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

# GPU slots, one line each: "<CUDA_VISIBLE_DEVICES value> <slots> <description>". GPUS picks devices (default all visible),
# SLOTS the runs per device (default by memory: 1 below 80 GB, e.g. a 32 GB MIG slice; 2 below 120 GB, e.g. H100 or
# RTX Pro 6000; else 3, e.g. H200 or B200). Memory comes from CUDA, which works where nvidia-smi reports
# [Insufficient Permissions]; a slice too small for two runs fails with an NVML INTERNAL ASSERT instead of an OOM message.
gpu_slots() {
  uv run python - "${GPUS:-all}" "${SLOTS:-auto}" <<'EOF'
import os, sys, torch
visible = os.environ.get("CUDA_VISIBLE_DEVICES")   # keep a pre-set list (MIG UUIDs): the index selects within it
ids = range(torch.cuda.device_count()) if sys.argv[1] == "all" else [int(g) for g in sys.argv[1].split(",")]
for i in ids:
    gb = torch.cuda.get_device_properties(i).total_memory / 1e9
    n = int(sys.argv[2]) if sys.argv[2] != "auto" else 1 if gb < 80 else 2 if gb < 120 else 3
    print(visible.split(",")[i] if visible else i, n, f"{torch.cuda.get_device_name(i)}, {gb:.0f} GB")
EOF
}

# One job = one trial of a lane's plan, run as its own kev.experiment study in runs/<lane>-t<k>, so trials spread over
# GPUs. A lane that already finished as one study (runs/<lane>, the older layout) counts as done. Rerunning skips
# finished jobs and restarts incomplete ones (the partial study is kept as runs/<job>.incomplete-<time>).
train() {
  mkdir -p $LOGS runs/h200-plans
  local jobs=() lane k
  for lane in $LANES; do
    local planned; planned=$(trials_planned $lane)
    if [ "$(trials_done $lane)" -ge "$planned" ]; then echo "lane $lane: finished as one study, skipping"; continue; fi
    for ((k = 0; k < planned; k++)); do
      local job=$lane-t$k
      if [ "$(trials_done $job)" -ge 1 ]; then echo "$job: finished, skipping"; continue; fi
      if [ -e runs/$job ]; then
        local keep; keep=runs/$job.incomplete-$(date +%Y%m%d-%H%M%S); mv runs/$job $keep; echo "$job: incomplete, moved to $keep"
      fi
      uv run python -c "import json, sys; json.dump([json.load(open(sys.argv[1]))[int(sys.argv[2])]], open(sys.argv[3], 'w'))" \
        experiments/$lane.json $k runs/h200-plans/$job.json
      jobs+=($job)
    done
  done
  if [ ${#jobs[@]} -eq 0 ]; then echo "nothing to train"; return 0; fi

  # slot list, first slots of every GPU before second ones: 4 jobs on 4 GPUs get one GPU each
  local lines=() slots=() line s dev n rest
  mapfile -t lines < <(gpu_slots)
  for line in "${lines[@]}"; do read -r dev n rest <<< "$line"; echo "GPU $dev: $rest -> $n run(s) at a time"; done
  for ((s = 0; s < 3; s++)); do
    for line in "${lines[@]}"; do
      read -r dev n rest <<< "$line"; if [ $s -lt $n ]; then slots+=("$dev"); fi
    done
  done
  echo "${#jobs[@]} runs over ${#slots[@]} slot(s): ${jobs[*]}"

  local -A pid_of job_of
  local failed=0 next=0 i pid job
  while :; do
    for i in "${!slots[@]}"; do
      pid=${pid_of[$i]:-}
      if [ -n "$pid" ] && ! kill -0 $pid 2>/dev/null; then   # bash reaps finished jobs and keeps their status for wait
        if wait $pid; then echo "$(date +%T) ${job_of[$i]} finished"; else echo "$(date +%T) ${job_of[$i]} failed; see $LOGS/${job_of[$i]}.log" >&2; failed=1; fi
        unset "pid_of[$i]"; pid=
      fi
      if [ -z "$pid" ] && [ $next -lt ${#jobs[@]} ]; then
        job=${jobs[$next]}; next=$((next + 1))
        CUDA_VISIBLE_DEVICES=${slots[$i]} uv run python -m kev.experiment --suite $SUITE --plan runs/h200-plans/$job.json \
          --transfer $TRANSFER --out runs/$job --queue $job --device cuda > $LOGS/$job.log 2>&1 &
        pid_of[$i]=$!; job_of[$i]=$job
        echo "$(date +%T) $job -> GPU ${slots[$i]} (log $LOGS/$job.log)"
        sleep ${STAGGER:-20}   # stagger model loading
      fi
    done
    if [ $next -ge ${#jobs[@]} ] && [ ${#pid_of[@]} -eq 0 ]; then break; fi
    sleep ${POLL:-30}
  done
  return $failed
}

evaluate() {
  # the winner: best transfer-v4 development accuracy over every finished trial (model selection on development data only)
  local winner
  export CUDA_VISIBLE_DEVICES; CUDA_VISIBLE_DEVICES=$(gpu_slots | head -1 | cut -d' ' -f1)   # evaluate on the first selected GPU
  winner=$(uv run python - $LANES <<'EOF'
import json, sys
from pathlib import Path
rows = []
for lane in sys.argv[1:]:
    studies = [Path("runs", lane)] + [p for p in Path("runs").glob(f"{lane}-t*") if ".incomplete" not in p.name]
    for result in sorted(r for study in studies for r in study.glob("*/result.json")):
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
      uv run python -m kev.compare --candidate runs/h200-eval/winner-$name --reference runs/h200-eval/$tag-$name --out runs/h200-eval/winner-vs-$tag-$name.json
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
