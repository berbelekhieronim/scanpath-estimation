#!/usr/bin/env bash
# Start/stop/restart the app for local development and in a Codespace.
#
#   tools/devserver.sh start [port]
#   tools/devserver.sh stop [port]
#   tools/devserver.sh restart [port]
#   tools/devserver.sh status [port]
#
# The bug this rewrite exists for: the old version recorded `$!` after
# `setsid nohup ... &`. Whether that is uvicorn's PID depends on whether
# setsid forks or execs, which depends on whether the shell made the
# subshell a process-group leader. When it forked, the PID file pointed at a
# process that had already exited — so `stop` found nothing to kill, happily
# deleted the PID file, and `start` launched a second server against a port
# the first one still held. The second died with "address already in use";
# the FIRST kept answering, running whatever code it had imported at boot.
#
# That is the whole "stale server" class of bug. The pages come off disk and
# update with a pull, the routes do not, and every symptom points at the
# code rather than at a process nobody knew was alive.
#
# So: the PID file is no longer trusted as the only truth. Whatever is
# actually listening on the port is the truth, and this script finds it.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${2:-${SCANPATH_PORT:-8000}}"
PIDFILE="$ROOT/.devserver-$PORT.pid"
LOG="$ROOT/.devserver-$PORT.log"

# Who is on the port? Several ways, because a slim container has none of
# them reliably: lsof is absent from many images, ss is absent from others,
# and fuser lives in psmisc which is not always installed.
listeners() {
  local port="$1"
  { lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null \
    || fuser "$port/tcp" 2>/dev/null \
    || ss -lptnH "sport = :$port" 2>/dev/null \
       | grep -oE 'pid=[0-9]+' | cut -d= -f2; } | tr -s ' \n' '\n' | sort -u
}

# Is that PID one of ours, or somebody else's service? Killing a stranger's
# process because it happened to want port 8000 would be its own bug.
is_ours() {
  local pid="$1" cmd
  cmd="$(ps -p "$pid" -o args= 2>/dev/null)" || return 1
  case "$cmd" in *"uvicorn app.main:app"*|*"app.main:app"*) return 0 ;; esac
  return 1
}

port_free() {
  [ -z "$(listeners "$1")" ]
}

describe_port() {
  local port="$1" pid
  for pid in $(listeners "$port"); do
    printf '  pid %s  %s\n' "$pid" "$(ps -p "$pid" -o args= 2>/dev/null | cut -c1-90)"
  done
}

stop() {
  local port="$1" killed=0 pid
  # Anything of ours on the port, whatever the PID file believes.
  for pid in $(listeners "$port"); do
    if is_ours "$pid"; then
      kill "$pid" 2>/dev/null && killed=1
    else
      echo "port $port is held by a process that is not this app:"
      describe_port "$port"
      echo "leaving it alone — pick another port, or stop it yourself"
      return 1
    fi
  done
  # And anything ours that lost its port but is still running, which is how
  # a half-dead server keeps the next start confusing.
  if [ -f "$PIDFILE" ]; then
    pid="$(cat "$PIDFILE" 2>/dev/null)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && is_ours "$pid"; then
      kill "$pid" 2>/dev/null && killed=1
    fi
    rm -f "$PIDFILE"
  fi

  # Give them a moment, then insist. A server that ignores SIGTERM and holds
  # the port is exactly the state this script has to be able to get out of.
  for _ in $(seq 1 20); do
    port_free "$port" && break
    sleep 0.25
  done
  if ! port_free "$port"; then
    for pid in $(listeners "$port"); do
      is_ours "$pid" && kill -9 "$pid" 2>/dev/null
    done
    sleep 0.5
  fi
  [ "$killed" = 1 ] && return 0 || return 0
}

start() {
  local port="$1"
  cd "$ROOT" || exit 1

  if ! port_free "$port"; then
    local mine=1 pid
    for pid in $(listeners "$port"); do is_ours "$pid" || mine=0; done
    if [ "$mine" = 1 ]; then
      echo "an older copy of this app is on port $port; replacing it"
      stop "$port" || return 1
    else
      echo "port $port is taken by something else:"
      describe_port "$port"
      # Not ours to kill, so move rather than fail. In a Codespace the
      # forwarded port then changes, which is worth saying out loud.
      local alt=$((port + 1))
      while [ "$alt" -lt $((port + 20)) ] && ! port_free "$alt"; do
        alt=$((alt + 1))
      done
      if [ "$alt" -ge $((port + 20)) ]; then
        echo "no free port in $port-$((port + 19)); stop something and retry"
        return 1
      fi
      echo "starting on port $alt instead"
      echo "in a Codespace: forward $alt in the Ports panel, and use that URL"
      port="$alt"
      PIDFILE="$ROOT/.devserver-$port.pid"
      LOG="$ROOT/.devserver-$port.log"
    fi
  fi

  setsid nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port "$port" \
    > "$LOG" 2>&1 < /dev/null &
  local spawned=$!

  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$port/healthz" >/dev/null 2>&1; then
      # The PID that is actually serving, not whatever the shell handed back:
      # setsid may have forked, in which case $! exited seconds ago.
      local real
      real="$(listeners "$port" | head -1)"
      echo "${real:-$spawned}" > "$PIDFILE"
      echo "up on port $port (pid ${real:-$spawned})"
      grep -E "control token" "$LOG" || true
      return 0
    fi
    # If it died, say why now rather than after the full timeout.
    if ! kill -0 "$spawned" 2>/dev/null && [ -z "$(listeners "$port")" ]; then
      break
    fi
    sleep 0.4
  done
  echo "failed to start on port $port; last lines of $LOG:"
  tail -20 "$LOG"
  return 1
}

status() {
  local port="$1" pid found=0
  for pid in $(listeners "$port"); do
    found=1
    if is_ours "$pid"; then
      echo "this app is running on port $port (pid $pid)"
      curl -sf "http://127.0.0.1:$port/healthz" >/dev/null 2>&1 \
        && echo "  healthz: ok" || echo "  healthz: NOT ANSWERING"
    else
      echo "port $port is held by something else:"
      describe_port "$port"
    fi
  done
  [ "$found" = 1 ] || echo "nothing is listening on port $port"
}

case "${1:-start}" in
  start)   start "$PORT" ;;
  stop)    stop "$PORT" && echo "stopped" ;;
  restart) stop "$PORT"; start "$PORT" ;;
  status)  status "$PORT" ;;
  *) echo "usage: $0 {start|stop|restart|status} [port]"; exit 2 ;;
esac
