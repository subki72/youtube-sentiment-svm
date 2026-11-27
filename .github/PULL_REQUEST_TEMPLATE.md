## Summary

<!-- Describe what this PR changes and why. -->

## Type of Change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Refactoring / performance improvement
- [ ] Documentation / CI / governance update

## Verification Checklist

- [ ] `python -m ruff check src tests main.py` passes with 0 errors
- [ ] `python -m ruff format --check src tests main.py` passes
- [ ] `python -m flake8 --max-line-length=120 src tests main.py` passes
- [ ] `python -m pytest tests/ -v --cov=src --cov-fail-under=75` passes 100% offline
- [ ] No secrets or `.env` files are staged
