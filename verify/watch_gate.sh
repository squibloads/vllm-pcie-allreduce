#!/usr/bin/env bash
# watch_gate.sh LOG : follow a verify_car.sh run from another shell. Streams each case's start/done line and
# every verdict, and every INTERVAL s (default 30) prints which car-* container is running, how many sizes
# it has reported and how long since its last one. STALL is flagged once a case adds no row for STALL_S s
# (default 600; the soak sizes take minutes each). car_gate.sh's CAP still stops a hung case; this only
# shows it sooner.
#   nohup bash -c 'IMG=your-1cat-image GPUS=0,1,2,3 verify/verify_car.sh; echo "gate rc=$?"' >gate.log 2>&1 &
#   verify/watch_gate.sh gate.log
log=${1:?usage: watch_gate.sh LOG}
interval=${INTERVAL:-30}
stall=${STALL_S:-600}
tail -n +1 -F "$log" 2>/dev/null | grep --line-buffered -E "^[0-9:]+ (start|done)|^(pass|FAIL)|gate rc" &
tp=$!
trap 'kill $tp 2>/dev/null' EXIT
cur="" rows=-1 since=$SECONDS flagged=""
while sleep "$interval"; do
  grep -q "gate rc" "$log" 2>/dev/null && break
  c=$(docker ps --format '{{.Names}}' | grep '^car-' | grep -vx 'car-load' | head -1)
  [ -z "$c" ] && continue
  n=$(docker logs "$c" 2>&1 | grep -c '^{')
  if [ "$c" != "$cur" ] || [ "$n" != "$rows" ]; then cur=$c rows=$n since=$SECONDS; fi
  idle=$((SECONDS - since))
  echo "$(date +%H:%M:%S) $c: $n sizes reported, last ${idle}s ago"
  if [ "$idle" -ge "$stall" ] && [ "$flagged" != "$c:$n" ]; then
    echo "STALL $c: no new size for ${idle}s"; flagged="$c:$n"
  fi
done
sleep 1
