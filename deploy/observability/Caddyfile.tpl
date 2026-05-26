{
  auto_https off
}

:${OBSERVABILITY_GATEWAY_PORT} {
  redir /grafana /grafana/ 308
  redir /prometheus /prometheus/ 308

  handle /grafana/* {
    reverse_proxy grafana:3000
  }

  handle /prometheus/* {
    reverse_proxy prometheus:9090
  }

  handle {
    reverse_proxy ${VLLM_API_TARGET}
  }
}
