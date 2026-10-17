# Deployment Guide

> **Build date:** 17 October 2026
> **Maintainer:** Adil Shamim (`@AdilShamim8`)

This guide walks through deploying Deep Research 104 to a production
environment. It covers local Docker, single-node VPS, and
Kubernetes-style deployments.

## Table of contents
- [Prerequisites](#prerequisites)
- [Quick deploy (Docker Compose)](#quick-deploy-docker-compose)
- [Single-node VPS deploy](#single-node-vps-deploy)
- [Kubernetes deploy](#kubernetes-deploy)
- [TLS termination](#tls-termination)
- [Monitoring](#monitoring)
- [Upgrades + rollbacks](#upgrades--rollbacks)
- [Troubleshooting](#troubleshooting)

## Prerequisites

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU       | 2 cores | 4 cores |
| RAM       | 2 GiB   | 4 GiB (+16 GiB if running local Ollama + R1:7b) |
| Disk      | 10 GiB  | 50 GiB (logs, cache, optional local model) |
| Docker    | 24.0    | 26.0+ |
| Compose   | v2.20+  | v2.24+ |
| OS        | Ubuntu 22.04 / Debian 12 / RHEL 9 | latest LTS |

## Quick deploy (Docker Compose)

```bash
git clone https://github.com/AdilShamim8/Deep-Research-104.git
cd Deep-Research-104

# 1. Copy the env template and fill in real values
cp .env.example .env
nano .env
# Set at minimum:
#   APP_ENV=production
#   JWT_SECRET=<32+ random chars>
#   OPENAI_API_KEY=sk-...
#   ALLOWED_ORIGINS=https://yourdomain.com

# 2. Bring up the stack (API + Redis)
make docker-up

# 3. Verify
curl http://localhost:8000/health
curl http://localhost:8000/version
```

To add the local LLM stack (Ollama + DeepSeek-R1):

```bash
make docker-up -- --profile local
```

To add Prometheus + Grafana:

```bash
make docker-up -- --profile monitoring
# Grafana: http://localhost:3000 (admin / $GRAFANA_ADMIN_PASSWORD)
```

## Single-node VPS deploy

The default `make docker-up` is sufficient for a single-node VPS
behind a TLS-terminating reverse proxy (nginx, Caddy, Traefik).

### nginx example

```nginx
server {
    listen 443 ssl http2;
    server_name deepresearch.example.com;

    ssl_certificate     /etc/letsencrypt/live/deepresearch.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/deepresearch.example.com/privkey.pem;

    # HSTS — the app also sets this header, but nginx should set it earlier
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Body size cap aligned with REQUEST_MAX_BODY_BYTES (1 MiB)
    client_max_body_size 1m;

    # Ingress rate limit (per-IP)
    limit_req_zone $binary_remote_addr zone=dr:10m rate=30r/m;

    location / {
        limit_req zone=dr burst=20 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;            # required for SSE
        proxy_read_timeout 600s;        # research can take minutes
    }

    # Lock down /metrics behind an auth subrequest or IP allowlist
    location /metrics {
        allow 10.0.0.0/8;
        deny  all;
        proxy_pass http://127.0.0.1:8000;
    }
}

server {
    listen 80;
    server_name deepresearch.example.com;
    return 301 https://$host$request_uri;
}
```

### Caddy example (auto-TLS)

```caddy
deepresearch.example.com {
    reverse_proxy 127.0.0.1:8000 {
        flush_interval -1   # required for SSE
    }
    request_body {
        max_size 1MB
    }
    header Strict-Transport-Security "max-age=31536000; includeSubDomains"
}
```

## Kubernetes deploy

A minimal `Deployment` + `Service` + `HorizontalPodAutoscaler` sketch:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deep-research-api
spec:
  replicas: 3
  selector:
    matchLabels: {app: deep-research-api}
  template:
    metadata:
      labels: {app: deep-research-api}
    annotations:
      prometheus.io/scrape: "true"
      prometheus.io/port: "8000"
      prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: api
          image: ghcr.io/adilshamim8/deep-research-104:1.0.0
          ports: [{containerPort: 8000}]
          envFrom:
            - secretRef: {name: deep-research-secrets}
            - configMapRef: {name: deep-research-config}
          readinessProbe:
            httpGet: {path: /ready, port: 8000}
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet: {path: /health, port: 8000}
            initialDelaySeconds: 15
            periodSeconds: 30
          resources:
            requests: {cpu: "500m", memory: "512Mi"}
            limits:   {cpu: "2000m", memory: "2Gi"}
---
apiVersion: v1
kind: Service
metadata:
  name: deep-research-api
spec:
  selector: {app: deep-research-api}
  ports: [{port: 80, targetPort: 8000}]
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: deep-research-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: deep-research-api
  minReplicas: 3
  maxReplicas: 50
  metrics:
    - type: Resource
      resource: {name: cpu, target: {type: Utilization, averageUtilization: 70}}
    - type: Resource
      resource: {name: memory, target: {type: Utilization, averageUtilization: 80}}
```

Build the image with the in-repo Dockerfile and push it to your
registry (GHCR, ECR, GCR, etc.) before applying.

## TLS termination

Always terminate TLS at the ingress / reverse proxy, not inside the
uvicorn process. The app sets `Strict-Transport-Security` for you,
and the `--proxy-headers` uvicorn flag makes it honor
`X-Forwarded-*` headers correctly.

For internal-only services (no public ingress), you can run the app
on plain HTTP behind a service mesh that handles mTLS.

## Monitoring

### Prometheus

The compose file already starts Prometheus with the in-repo
`config/prometheus.yml`. To scrape the app from an external
Prometheus, point a scrape job at `host:8000/metrics`.

Suggested alerts:

```yaml
groups:
  - name: deep-research
    rules:
      - alert: DeepResearchHighErrorRate
        expr: |
          rate(deep_research_requests_total{status_code=~"5.."}[5m])
          / rate(deep_research_requests_total[5m]) > 0.05
        for: 10m
        annotations:
          summary: "Error rate above 5% for 10m"
      - alert: DeepResearchHighLatency
        expr: |
          histogram_quantile(0.95,
            rate(deep_research_request_duration_seconds_bucket[5m]))
          > 60
        for: 5m
        annotations:
          summary: "p95 latency above 60s for 5m"
      - alert: DeepResearchNoCacheHits
        expr: |
          deep_research_cache_hits_total == 0
        for: 15m
        annotations:
          summary: "No cache hits in 15m — Redis may be down"
```

### Grafana

The compose file starts Grafana with no pre-provisioned dashboards.
Connect it to the local Prometheus datasource and import the
included metric names (see `ARCHITECTURE.md` §5).

### Logs

In production, logs are JSON to stdout (one event per line) plus
rotated files under `logs/`. Ship stdout to your log aggregator
(Loki, ELK, Datadog, CloudWatch). Each line carries `request_id` so
you can grep for all events in a single user request.

## Upgrades + rollbacks

### Rolling upgrade (Docker Compose)

```bash
git pull
make docker-up     # compose pulls the new image and restarts
```

The app starts in <5 seconds. Health checks block the old container
from being killed until the new one is healthy.

### Rollback

```bash
git checkout <previous-tag>
make docker-up
```

Or, if you're using a registry:

```bash
docker pull ghcr.io/adilshamim8/deep-research-104:<previous-version>
docker-compose up -d --no-deps api
```

### Database migrations

1.0.0 doesn't ship a database. When `StorageConfig.storage_backend`
is `postgres`, an Alembic migration runner will be added in 1.1.

## Troubleshooting

| Symptom | First check |
|---------|-----------|
| 503 on `/ready` | `kubectl logs` / `docker logs` — check `subsystem_status` |
| 500 on `/research` | Look for `error_code` in the JSON response body |
| High 429 from providers | Increase `RATE_LIMIT_RPM` for the affected model in `.env` |
| Empty answer | Search provider down — check `/metrics/search` for `requests_failed` |
| High token cost | Check `deep_research_tokens_total` in Prometheus; consider lowering `MAX_RESEARCH_ITERATIONS` |
| OOM kills | Lower `--workers`, reduce `MAX_CONTEXT_TOKENS`, or scale vertically |
| SSE buffering | Set `proxy_buffering off;` in nginx (or `flush_interval -1` in Caddy) |
| /metrics returning 404 | You're hitting the route under `/api/v1`; the real metrics endpoint is at the root: `/metrics` |
