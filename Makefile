.PHONY: install lint format typecheck test health demo serve docker-build docker-up clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check src tests main.py
	$(PYTHON) -m ruff format --check src tests main.py
	$(PYTHON) -m flake8 --max-line-length=120 src tests main.py

format:
	$(PYTHON) -m ruff check --fix src tests main.py
	$(PYTHON) -m ruff format src tests main.py

typecheck:
	$(PYTHON) -m mypy src main.py --ignore-missing-imports

test:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=75

health:
	$(PYTHON) main.py --health

demo:
	$(PYTHON) main.py --demo

serve:
	$(PYTHON) main.py --serve

docker-build:
	docker build -t youtube-sentiment-svm:latest .

docker-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml build dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
