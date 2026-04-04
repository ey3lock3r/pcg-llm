# Makefile for pcg-llm development
# Usage: make <target>

.PHONY: help install install-pre-commit test test-fast test-verbose lint lint-fix \
        format format-check typecheck security validate-specs check ci clean \
        coverage-html pre-commit update-deps

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv pip install -e ".[dev]"
	@echo "✅ Dependencies installed"

install-pre-commit:  ## Install pre-commit hooks
	uv pip install pre-commit
	pre-commit install
	@echo "✅ Pre-commit hooks installed"

test:  ## Run tests with coverage
	pytest tests/ -v --cov=src/pcg_llm --cov-report=term-missing

test-fast:  ## Run tests without coverage (faster)
	pytest tests/ -v -p no:cov

test-verbose:  ## Run tests with maximum verbosity
	pytest tests/ -vv -s

test-convergence:  ## Run only convergence property tests
	pytest tests/ -v -m convergence

lint:  ## Run linting checks
	@echo "Running Ruff linter..."
	ruff check src/ tests/

lint-fix:  ## Run linting with auto-fix
	@echo "Running Ruff linter with auto-fix..."
	ruff check --fix src/ tests/

format:  ## Format code
	@echo "Formatting code with Ruff..."
	ruff format src/ tests/
	@echo "✅ Code formatted"

format-check:  ## Check if code is formatted correctly
	@echo "Checking code formatting..."
	ruff format --check src/ tests/

typecheck:  ## Run type checking
	@echo "Running mypy type checker..."
	mypy src/pcg_llm

security:  ## Run security checks
	@echo "Running Bandit security scan..."
	bandit -r src/ -c pyproject.toml

validate-specs:  ## Validate specification files
	@echo "Validating specs..."
	python scripts/validate_specs.py

check: lint typecheck test  ## Run all checks (lint, typecheck, test)
	@echo "✅ All checks passed!"

ci:  ## Run full CI pipeline locally
	@echo "================================"
	@echo "Running CI Pipeline Locally"
	@echo "================================"
	@echo ""
	@echo "Step 1: Linting..."
	@$(MAKE) lint
	@echo ""
	@echo "Step 2: Format checking..."
	@$(MAKE) format-check
	@echo ""
	@echo "Step 3: Type checking..."
	@$(MAKE) typecheck
	@echo ""
	@echo "Step 4: Security scanning..."
	@$(MAKE) security
	@echo ""
	@echo "Step 5: Validating specs..."
	@$(MAKE) validate-specs
	@echo ""
	@echo "Step 6: Running tests..."
	@$(MAKE) test
	@echo ""
	@echo "================================"
	@echo "✅ CI Pipeline Passed!"
	@echo "================================"

clean:  ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned"

coverage-html:  ## Generate HTML coverage report
	pytest tests/ --cov=src/pcg_llm --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

pre-commit:  ## Run pre-commit on all files
	pre-commit run --all-files

update-deps:  ## Update dependencies
	uv pip install --upgrade -e ".[dev]"
	@echo "✅ Dependencies updated"

.DEFAULT_GOAL := help
