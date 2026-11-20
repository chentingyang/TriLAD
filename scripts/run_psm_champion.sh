#!/usr/bin/env bash
# PSM VLM-AD champion reproduction entry
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[[ -f .env ]] && set -a && source .env && set +a
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
PY="${PYTHON:-python3}"
ACTION="${1:-run}"
shift || true
case "$ACTION" in
  show|run|score)
    exec "$PY" -u scripts/champion.py psm "$ACTION" "$@"
    ;;
  -h|--help)
    echo "Usage: $0 {show|run|score} [champion.py options]"
    echo "  show   Print PSM champion reproduction config"
    echo "  run    Fresh API inference + offline score (default; all windows)"
    echo "  score  Offline score an existing run (--run_dir or --tag)"
    echo "Examples:"
    echo "  $0 show"
    echo "  $0 run --concurrency 8"
    echo "  $0 score --run_dir runs/psm/<tag>"
    exit 0
    ;;
  *)
    echo "Unknown action: $ACTION (use show|run|score)" >&2
    exit 1
    ;;
esac
