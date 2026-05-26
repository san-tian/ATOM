global:
  scrape_interval: ${VLLM_SCRAPE_INTERVAL}
  evaluation_interval: ${VLLM_SCRAPE_INTERVAL}

scrape_configs:
  - job_name: vllm
    scheme: ${VLLM_METRICS_SCHEME}
    metrics_path: ${VLLM_METRICS_PATH}
    static_configs:
      - targets:
          - ${VLLM_METRICS_TARGET}
        labels:
          service: vllm

  - job_name: prometheus
    static_configs:
      - targets:
          - localhost:9090
