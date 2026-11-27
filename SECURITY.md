# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.0.x   | Yes       |
| < 1.0   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in this repository:

1. **Do not open a public GitHub issue** for undisclosed vulnerabilities.
2. Submit a private report via GitHub Security Advisories (`Security -> Advisories -> Report a vulnerability`).
3. Include affected file paths, line numbers, reproduction steps, and estimated impact.
4. Initial acknowledgement will be provided within **48 hours**.

## Security Controls

- **Secret Hygiene**: `.env` and `.env.*` files are excluded via `.gitignore`; only `.env.example` placeholders are committed.
- **Network Safety**: Remote lexicon downloads are disabled by default (`allow_remote=False`) and enforce explicit HTTP timeouts (`timeout=5.0`s) plus `raise_for_status()`.
- **API Hardening**: Pydantic v2 input validation, per-client sliding-window rate limiting, CORS origin filtering, and sanitized 500 error responses without tracebacks.
- **Automated Vulnerability Management**: Weekly Dependabot alerts (`semver-major` ignored), Semgrep SAST, and Trivy container image scanning in CI.
- **Container Hardening**: Multi-stage non-root Docker builds running as `appuser`.
