#!/usr/bin/env bash
# car_gate.sh MODE [TAG] : run car_gate.py in your 1Cat-vLLM image, one detached container, at most CAP s.
#   MODE  nccl | car | car-nopush | negctl | perturb   (car_gate.py explains each)
#   TAG   names the case (default MODE); the full log goes to $OUT/car.TAG.log and ends with a line
#         "car_gate.sh: exit=<container exit code> timeout=<0|1> load=<car_load summary|none|MISSING>"
# Environment:
#   IMG       your 1Cat-vLLM image (required)
#   GPUS      the four GPUs under test, as `docker run --gpus device=` takes them: nvidia-smi indices or
#             UUIDs (default 0,1,2,3). Indices follow PCI bus order, which moves when cards or switches
#             move; UUIDs do not.
#   SET       main (default) runs on GPUS; ref runs on REF_GPUS, the same seeds on another set or order
#   REF_GPUS  default: GPUS reversed, so every rank lands on a different GPU and path
#   LOAD=1    car_load.py drives copy-engine traffic between LOAD_GPUS (default GPUS) meanwhile
#   OUT       log directory (default ./car-results)
#   CAP       seconds before the case is stopped and reported as a timeout (default 1200)
#   DOCKER_USER  user:group inside the containers (default: yours)
# car_gate.py's own knobs (DATA, SIZES, EAGER_ITERS, CHAIN, CHAIN_REPLAYS, SKEW, SOAK, SOAK_SIZES, TIME_K,
# TIME_REPLAYS, PROFILE) pass through from the environment.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mode=${1:?usage: car_gate.sh nccl|car|car-nopush|negctl|perturb [TAG]} tag=${2:-$1}
IMG=${IMG:?set IMG to your 1Cat-vLLM image}
GPUS=${GPUS:-0,1,2,3}
IFS=, read -ra g <<<"$GPUS"
[ "${#g[@]}" -eq 4 ] || { echo "GPUS must name exactly four GPUs, got '$GPUS'" >&2; exit 2; }
rev=""; for ((i = ${#g[@]} - 1; i >= 0; i--)); do rev+="${g[i]},"; done
REF_GPUS=${REF_GPUS:-${rev%,}}
case "${SET:-main}" in
  main) G=$GPUS;;
  ref) G=$REF_GPUS;;
  *) echo "unknown SET '${SET}' (main|ref)" >&2; exit 2;;
esac
OUT=${OUT:-car-results}
mkdir -p "$OUT"
cap=${CAP:-1200}
user=${DOCKER_USER:-$(id -u):$(id -g)}
pass=()
for k in DATA SIZES EAGER_ITERS CHAIN CHAIN_REPLAYS SKEW SOAK SOAK_SIZES TIME_K TIME_REPLAYS PROFILE; do
  [ -n "${!k:-}" ] && pass+=(-e "$k=${!k}")
done
name=car-$tag
docker rm -f "$name" car-load >/dev/null 2>&1
load=none
if [ "${LOAD:-0}" = 1 ]; then
  load=MISSING
  docker run -d --name car-load --gpus "\"device=${LOAD_GPUS:-$GPUS}\"" -u "$user" -e HOME=/tmp \
    -v "$HERE:/w:ro" --entrypoint python3 "$IMG" /w/car_load.py "$cap" >/dev/null
fi
docker run -d --name "$name" --gpus "\"device=$G\"" --ipc=host -u "$user" \
  -e HOME=/tmp -e NCCL_P2P_LEVEL=SYS -e VLLM_CACHE_ROOT=/tmp/vllm ${pass[@]+"${pass[@]}"} \
  -v "$HERE:/w:ro" --entrypoint python3 "$IMG" /w/car_gate.py "$mode" >/dev/null
t=0
while [ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" = true ] && [ $t -lt "$cap" ]; do
  sleep 2; t=$((t + 2))
done
timeout=0
if [ $t -ge "$cap" ]; then
  timeout=1; echo "TIMEOUT $name after ${cap}s (stopped)"; docker stop -t 2 "$name" >/dev/null
fi
rc=$(docker inspect -f '{{.State.ExitCode}}' "$name" 2>/dev/null || echo unknown)
if [ "${LOAD:-0}" = 1 ]; then
  docker stop -t 2 car-load >/dev/null 2>&1
  s=$(docker logs car-load 2>&1 | grep -m1 '^car_load:')
  [ -n "$s" ] && load=${s#car_load: }
  docker rm car-load >/dev/null 2>&1
fi
{
  docker logs "$name" 2>&1 | head -c 20000000
  echo "car_gate.sh: exit=$rc timeout=$timeout load=$load"
} >"$OUT/car.$tag.log"
docker rm "$name" >/dev/null 2>&1
grep -E '^\{' "$OUT/car.$tag.log" || grep -E 'Error|error|Traceback' "$OUT/car.$tag.log" | tail -5
tail -1 "$OUT/car.$tag.log"
