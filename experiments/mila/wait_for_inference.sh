#!/usr/bin/env bash
set -euo pipefail

HOST0="${1:?Usage: wait_for_inference.sh <host0> <host1> [port] [timeout_seconds]}"
HOST1="${2:?Usage: wait_for_inference.sh <host0> <host1> [port] [timeout_seconds]}"
PORT="${3:-8000}"
TIMEOUT="${4:-900}"

probe() {
  local host="$1"
  curl -fsS --max-time 5 "http://${host}:${PORT}/v1/models" >/dev/null 2>&1
}

echo "Waiting for inference servers on ${HOST0}:${PORT} and ${HOST1}:${PORT}..."
elapsed=0
ready0=0
ready1=0

while (( elapsed < TIMEOUT )); do
  if (( ready0 == 0 )) && probe "$HOST0"; then
    ready0=1
    echo "  ${HOST0} ready at ${elapsed}s"
  fi
  if (( ready1 == 0 )) && probe "$HOST1"; then
    ready1=1
    echo "  ${HOST1} ready at ${elapsed}s"
  fi
  if (( ready0 == 1 && ready1 == 1 )); then
    echo "Both inference servers are ready."
    exit 0
  fi
  sleep 10
  elapsed=$((elapsed + 10))
  echo "  waiting... (${elapsed}/${TIMEOUT}s) [host0=${ready0} host1=${ready1}]"
done

echo "ERROR: inference servers were not ready within ${TIMEOUT}s"
exit 1
