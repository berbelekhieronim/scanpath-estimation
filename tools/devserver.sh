#!/usr/bin/env bash
# Start/stop/restart the app for local development.
#   tools/devserver.sh start [port]
#   tools/devserver.sh stop
#   tools/devserver.sh restart [port]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${2:-8000}"
PIDFILE="$ROOT/.devserver.pid"
LOG="$ROOT/.devserver.log"

stop() {
  if [ -f "$PIDFILE" ]; then
    pid="$(cat "$PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null; sleep 1; fi
    rm -f "$PIDFILE"
  fi
}

start() {
  cd "$ROOT" || exit 1
  setsid nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" \
    > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 40); do
    if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
      echo "up on port $PORT"; grep -E "control token" "$LOG"; return 0
    fi
    sleep 0.4
  done
  echo "failed to start; last lines of $LOG:"; tail -15 "$LOG"; return 1
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop; echo "stopped" ;;
  restart) stop; start ;;
  *) echo "usage: $0 {start|stop|restart} [port]"; exit 2 ;;
esac
