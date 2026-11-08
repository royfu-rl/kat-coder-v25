#!/usr/bin/env bash
# KAT-Coder blog — Linux / AutoDL dev server.
#
#   ./serve-autodl.sh              build + serve on :6006 (AutoDL custom-service port)
#   ./serve-autodl.sh 8000         use a different port
#   ./serve-autodl.sh --no-build   skip rebuild, just (re)serve
#   ./serve-autodl.sh stop         stop the running server
#
# Binds 0.0.0.0 so AutoDL's "自定义服务" tunnel (default port 6006) can reach it.
# Idempotent: re-running kills any previous instance on the same port first.
set -euo pipefail

cd "$(dirname "$0")"

PY="$(command -v python3 || command -v python)"
PORT=6006
BUILD=1
PIDFILE="/tmp/kat-blog-serve.pid"
LOG="/tmp/kat-blog-serve.log"

for arg in "$@"; do
  case "$arg" in
    stop)
      if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        kill "$(cat "$PIDFILE")" && echo ">> stopped server (pid $(cat "$PIDFILE"))"
        rm -f "$PIDFILE"
      else
        echo ">> no running server found"
      fi
      exit 0 ;;
    --no-build) BUILD=0 ;;
    [0-9]*)     PORT="$arg" ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# Kill any previous instance we started (idempotent restart).
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo ">> stopping previous server (pid $(cat "$PIDFILE")) ..."
  kill "$(cat "$PIDFILE")" 2>/dev/null || true
  sleep 1
fi
rm -f "$PIDFILE"

if [ "$BUILD" -eq 1 ]; then
  echo ">> building index.html ..."
  "$PY" build.py
fi

echo ">> serving http://0.0.0.0:${PORT}/index.html  (log: ${LOG})"
nohup "$PY" -m http.server "$PORT" --bind 0.0.0.0 > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
sleep 2

# Health check.
if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/index.html" | grep -q 200; then
  echo ">> OK — server running (pid $(cat "$PIDFILE"))"
  echo ">> AutoDL: open the 自定义服务 URL (maps to port 6006)."
  echo ">> or SSH tunnel from your laptop:  ssh -L 8000:localhost:${PORT} root@<host> -p <ssh-port>"
else
  echo ">> WARN — health check failed, see ${LOG}:" >&2
  tail -20 "$LOG" >&2
  exit 1
fi

