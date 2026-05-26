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
: "${OBSERVABILITY_SINGLE_PORT:=false}"
: "${OBSERVABILITY_GATEWAY_PORT:=7777}"
: "${VLLM_API_TARGET:=$VLLM_METRICS_TARGET}"

mkdir -p "$ROOT_DIR/prometheus"

sed \
  -e "s|\${VLLM_METRICS_TARGET}|${VLLM_METRICS_TARGET}|g" \
  -e "s|\${VLLM_METRICS_SCHEME}|${VLLM_METRICS_SCHEME}|g" \
  -e "s|\${VLLM_METRICS_PATH}|${VLLM_METRICS_PATH}|g" \
  -e "s|\${VLLM_SCRAPE_INTERVAL}|${VLLM_SCRAPE_INTERVAL}|g" \
  "$ROOT_DIR/prometheus/prometheus.yml.tpl" > "$ROOT_DIR/prometheus/prometheus.yml"

COMPOSE_FILES=(-f "$ROOT_DIR/docker-compose.yml")
if [[ "$OBSERVABILITY_SINGLE_PORT" == "1" || "$OBSERVABILITY_SINGLE_PORT" == "true" || "$OBSERVABILITY_SINGLE_PORT" == "yes" ]]; then
  sed \
    -e "s|\${OBSERVABILITY_GATEWAY_PORT}|${OBSERVABILITY_GATEWAY_PORT}|g" \
    -e "s|\${VLLM_API_TARGET}|${VLLM_API_TARGET}|g" \
    "$ROOT_DIR/Caddyfile.tpl" > "$ROOT_DIR/Caddyfile"
  COMPOSE_FILES+=(-f "$ROOT_DIR/docker-compose.gateway.yml")
else
  COMPOSE_FILES+=(-f "$ROOT_DIR/docker-compose.ports.yml")
fi

docker compose --env-file "$ENV_FILE" "${COMPOSE_FILES[@]}" up -d

cat <<EOF
Observability stack is starting.
Dashboard:  ATOM / ATOM vLLM Overview
EOF

if [[ "$OBSERVABILITY_SINGLE_PORT" == "1" || "$OBSERVABILITY_SINGLE_PORT" == "true" || "$OBSERVABILITY_SINGLE_PORT" == "yes" ]]; then
  cat <<EOF
Single-port gateway: http://127.0.0.1:${OBSERVABILITY_GATEWAY_PORT}
Inference API:       http://127.0.0.1:${OBSERVABILITY_GATEWAY_PORT}/v1/models
vLLM metrics:        http://127.0.0.1:${OBSERVABILITY_GATEWAY_PORT}/metrics
Grafana:             http://127.0.0.1:${OBSERVABILITY_GATEWAY_PORT}/grafana/
Prometheus:          http://127.0.0.1:${OBSERVABILITY_GATEWAY_PORT}/prometheus/
EOF
else
  cat <<EOF
Prometheus: http://127.0.0.1:${PROMETHEUS_PORT:-9090}
Grafana:    http://127.0.0.1:${GRAFANA_PORT:-3000}
EOF
fi
