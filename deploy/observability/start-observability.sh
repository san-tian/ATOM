#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-$ROOT_DIR/.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ROOT_DIR/.env.example" "$ENV_FILE"
  echo "Created $ENV_FILE from .env.example. Edit it if the vLLM metrics target is not host.docker.internal:7791."
fi

set -a
source "$ENV_FILE"
set +a

: "${VLLM_METRICS_TARGET:=host.docker.internal:7791}"
: "${VLLM_METRICS_SCHEME:=http}"
: "${VLLM_METRICS_PATH:=/metrics}"
: "${VLLM_SCRAPE_INTERVAL:=5s}"

mkdir -p "$ROOT_DIR/prometheus"

sed \
  -e "s|\${VLLM_METRICS_TARGET}|${VLLM_METRICS_TARGET}|g" \
  -e "s|\${VLLM_METRICS_SCHEME}|${VLLM_METRICS_SCHEME}|g" \
  -e "s|\${VLLM_METRICS_PATH}|${VLLM_METRICS_PATH}|g" \
  -e "s|\${VLLM_SCRAPE_INTERVAL}|${VLLM_SCRAPE_INTERVAL}|g" \
  "$ROOT_DIR/prometheus/prometheus.yml.tpl" > "$ROOT_DIR/prometheus/prometheus.yml"

docker compose --env-file "$ENV_FILE" -f "$ROOT_DIR/docker-compose.yml" up -d

cat <<EOF
Observability stack is starting.
Prometheus: http://127.0.0.1:${PROMETHEUS_PORT:-9090}
Grafana:    http://127.0.0.1:${GRAFANA_PORT:-3000}
Dashboard:  ATOM / ATOM vLLM Overview
EOF
