# ATOM vLLM Observability

This template starts a small Prometheus + Grafana stack that scrapes a vLLM
OpenAI-compatible server through its Prometheus endpoint:

```text
http://<vllm-host>:<vllm-port>/metrics
```

It is parameterized so it can be started before the final host IP is known.
When the inference service is running on the same machine as Docker, the default
target is `host.docker.internal:7791`.

## Start

```bash
cd deploy/observability
cp .env.example .env

# Change this when vLLM is not bound to host port 7791.
vim .env

./start-observability.sh
```

Open:

- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`
- Default Grafana login: `admin` / `admin`
- Dashboard: `ATOM / ATOM vLLM Overview`

## Single External Port with Caddy

When only one external port is available, enable the Caddy gateway. This keeps
Prometheus and Grafana internal to Docker and publishes only one host port,
defaulting to `7777`.

```bash
cd deploy/observability
cp .env.example .env

cat >> .env <<'EOF'
OBSERVABILITY_SINGLE_PORT=true
OBSERVABILITY_GATEWAY_PORT=7777
VLLM_API_TARGET=host.docker.internal:7791
VLLM_METRICS_TARGET=host.docker.internal:7791
GRAFANA_ROOT_URL=http://<host-ip>:7777/grafana/
PROMETHEUS_EXTERNAL_URL=http://<host-ip>:7777/prometheus/
EOF

./start-observability.sh
```

External routes:

- vLLM OpenAI-compatible API: `http://<host-ip>:7777/v1/models`
- vLLM metrics: `http://<host-ip>:7777/metrics`
- Grafana: `http://<host-ip>:7777/grafana/`
- Prometheus: `http://<host-ip>:7777/prometheus/`

The vLLM server still listens on its local service port, for example `7791`.
Caddy is the only service that needs to bind an externally reachable host port.

## Connect a vLLM Service

The only required vLLM side is that the server exposes `/metrics` on its HTTP
port. For the verified ATOM + vLLM launcher, this is the same port used by
OpenAI-compatible APIs, for example `7791`.

Common `.env` values:

```bash
# vLLM runs on the Docker host and listens on 7791.
VLLM_METRICS_TARGET=host.docker.internal:7791

# vLLM runs in the same compose network as this stack.
VLLM_METRICS_TARGET=vllm:8000

# vLLM runs on another host.
VLLM_METRICS_TARGET=10.0.0.12:7791
```

After editing `.env`, rerun:

```bash
./start-observability.sh
```

## Check the Pipeline

```bash
curl -fsS "http://127.0.0.1:${PROMETHEUS_PORT:-9090}/-/ready"
curl -fsS "http://127.0.0.1:${PROMETHEUS_PORT:-9090}/api/v1/targets" | jq .
```

If Prometheus shows the vLLM target as down, first test the metrics endpoint
from the host:

```bash
curl -fsS "http://127.0.0.1:7791/metrics" | head
```

Then test from inside the Prometheus container:

```bash
docker compose exec prometheus wget -qO- \
  "http://${VLLM_METRICS_TARGET:-host.docker.internal:7791}${VLLM_METRICS_PATH:-/metrics}" \
  | head
```

## Dashboard Coverage

The bundled Grafana dashboard focuses on the signals that are useful while
running load tests:

- running and waiting requests
- successful request rate
- prompt and generation token throughput
- time-to-first-token and time-per-output-token percentiles
- vLLM KV cache usage

For host CPU, memory, disk, and GPU telemetry, add node exporter and the
ROCm/DCGM-compatible exporter used by your environment. This template keeps
the first version focused on vLLM's native `/metrics` endpoint so it can be
started on any inference host without extra kernel or ROCm permissions.

## Lighter Alternatives

| Option | What you get | Tradeoff |
|---|---|---|
| Prometheus UI only | Lowest moving parts. Start only Prometheus and use PromQL directly. | No persistent dashboards or sharing. Good for quick checks. |
| Netdata | Fast host-level dashboard for CPU, memory, disk, network, and GPU if the host plugin supports it. | Less natural for vLLM request/token metrics unless you still scrape Prometheus metrics. |
| OpenObserve | Single binary/service for logs, metrics, and traces. Useful when you also want request logs. | More configuration than Prometheus UI for simple `/metrics` scraping, and the Grafana ecosystem is smaller. |
| Prometheus + Grafana | Best default for vLLM metrics, dashboards, sharing, and alerting. | Two services and persistent volumes to manage. |

## Use an Existing Grafana

If Grafana is already deployed, run only Prometheus or point that Grafana at
this stack's Prometheus URL:

```text
http://<observability-host>:9090
```

Then import `grafana/dashboards/vllm-overview.json` or copy the provisioning
directory into your Grafana deployment.
