#!/usr/bin/env bash
# One-command dev server for the KAT-Coder blog.
#
#   ./serve.sh            build + serve on :8000, open browser
#   ./serve.sh 9000       use port 9000
#   ./serve.sh --no-build just serve (skip rebuild)
#
# Why a server (not double-click): deepswe/index.html loads ES modules and
# fetch()es JSON from artifacts/. Both are blocked under file://, so the
# embedded Visual Analysis Tool renders blank. Serving over HTTP fixes it.
set -euo pipefail

cd "$(dirname "$0")"

# Anaconda python — the bare `python` on PATH is a Windows Store stub that
# does nothing in a non-interactive shell.
PY="/c/Users/roy/anaconda3/python.exe"
[ -x "$PY" ] || PY="python"

PORT=8000
BUILD=1
for arg in "$@"; do
  case "$arg" in
    --no-build) BUILD=0 ;;
    [0-9]*)     PORT="$arg" ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

if [ "$BUILD" -eq 1 ]; then
  echo ">> building index.html ..."
  "$PY" build.py
fi

URL="http://localhost:${PORT}/index.html"
echo ">> serving on ${URL}  (Ctrl-C to stop)"

# Open the browser shortly after the server comes up (background).
( sleep 1; start "" "$URL" 2>/dev/null || true ) &

exec "$PY" -m http.server "$PORT"

