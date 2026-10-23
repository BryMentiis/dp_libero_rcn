#!/bin/bash
# GPU / CPU / MEM sampler — auto-detects the LIBERO training process
LOG=/root/autodl-tmp/gpu_monitor.log
while true; do
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits)
  PID=$(pgrep -f "lifelong/main[.]py" | head -1)
  if [ -n "$PID" ]; then
    CPU=$(ps -p "$PID" -o %cpu= | tr -d ' ')
  else
    CPU="DONE"
  fi
  MEM=$(free -g | awk '/Mem:/{print $3"G/"$2"G"}')
  echo "$TS | GPU_util:${GPU}%  | train_cpu%:${CPU} | mem:${MEM}" >> "$LOG"
  [ "$CPU" = "DONE" ] && echo "$TS | training process finished" >> "$LOG" && break
  sleep 60
done
