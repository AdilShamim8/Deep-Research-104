# Security Policy

## Supported versions

| Version | Supported | Notes |
|---------|-----------|-------|
| 1.0.x   | ✅       | Active development branch (main) |
| < 1.0   | ❌       | Pre-release; no security backports |

## Reporting a vulnerability

**Do NOT open a public GitHub issue for security vulnerabilities.**

Email **adilshamim8@users.noreply.github.com** with:

1. A description of the issue and its impact.
2. A minimal reproduction (steps, payload, or script).
3. The affected version (check `app.version` via `/version` or `pip show deep-research-104`).
4. Any mitigations you've already tried.

### Response SLAs

| Step | Target |
|------|--------|
| Acknowledge receipt | within 48 hours |
| Initial assessment + severity | within 5 business days |
| Fix or mitigation | within 14 days for High, 30 days for Medium |
| Public disclosure | after a fix is released, with credit if desired |

Please do not disclose the issue publicly until we have shipped a fix
or explicitly agreed on a disclosure timeline.

## Threat model

Deep Research 104 accepts user-supplied questions, runs them through
LLM providers, and returns synthesized reports. The primary assets at
risk are:

- **API keys / credentials** in `.env` or env vars (mitigated: never
  logged, never returned in responses, validated at startup).
- **User question content** (mitigated: not persisted by default;
  search results cached with TTL).
- **Outbound LLM tokens** (mitigated: rate-limited per model; cost
  tracking surfaced via `/metrics`).

Areas that warrant particular scrutiny when reviewing PRs:

| Area | Risk | Mitigation in place |
|------|------|---------------------|
| `src/api/routes.py` | Input validation bypass | `_StrictRequest` rejects unknown fields; field max_length caps |
| `src/search/web_search.py` | SSRF via search URL | `_is_safe_url` blocks `file://`, `ftp://`, etc. |
| `src/search/content_extractor.py` | Fetching arbitrary URLs | `BLOCKED_DOMAINS` + `BLOCKED_EXTENSIONS` + 2 MiB body cap |
| `src/models/*.py` | API key leakage | Keys live only on the provider client; never logged |
| `config/settings.py` | Misconfigured prod | `_validate_production_requirements` fails loudly on startup |

## Security headers shipped by default

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains` (production only)

## Hardening checklist (for self-hosters)

- [ ] Set `APP_ENV=production` and a 16+ char `JWT_SECRET`.
- [ ] Set `ALLOWED_ORIGINS` to your specific domains (never `*`).
- [ ] Set at least one of `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`.
- [ ] Put the API behind TLS (HSTS only matters over HTTPS).
- [ ] Add an ingress rate limit in front of the app (e.g. nginx
  limit_req_zone) — the in-process limiter is per-worker.
- [ ] Mount a read-only `prometheus.yml` and protect `/metrics` at
  the ingress (don't expose internal counts publicly).
- [ ] Run `make security` (pip-audit) in CI to catch advisories.
- [ ] Subscribe to GitHub Dependabot alerts for the repo.

## Disclosure credits

We thank responsible reporters and credit them (with permission) in
`CHANGELOG.md` once a fix is shipped.
