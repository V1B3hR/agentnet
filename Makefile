# AgentNet Development Makefile
# Phase 0 Development Utilities

.PHONY: help install install-dev install-full clean format lint test test-p0 build dist publish cli-format cli-lint cli-test

help:  ## Show this help message
	@echo "AgentNet Development Commands:"
	@echo "==============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install core dependencies (Phase 0)
	pip install pydantic pyyaml typing-extensions

install-dev:  ## Install development dependencies
	pip install pytest black isort flake8 mypy coverage click

install-full:  ## Install all dependencies (all phases)
	pip install -e ".[all]"

clean:  ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete

format:  ## Format code with black and isort (DEPRECATED: use cli-format)
	@echo "⚠️  DEPRECATED: Use 'make cli-format' instead"
	python scripts/format.py

cli-format:  ## Format code with black and isort (unified CLI)
	python -m cli.main format

lint:  ## Lint code with flake8 (DEPRECATED: use cli-lint)
	@echo "⚠️  DEPRECATED: Use 'make cli-lint' instead"
	python scripts/lint.py

cli-lint:  ## Lint code with flake8 (unified CLI)
	python -m cli.main lint

test:  ## Run all tests (DEPRECATED: use cli-test)
	@echo "⚠️  DEPRECATED: Use 'make cli-test' instead"
	python scripts/test.py

cli-test:  ## Run all tests (unified CLI)
	python -m cli.main test

test-p0:  ## Run Phase 0 tests only
	PYTHONPATH=. python -m pytest tests/test_p0_implementation.py -v

test-minimal:  ## Run tests that don't require async or network
	PYTHONPATH=. python -m pytest tests/test_p0_implementation.py tests/test_direct_module_import.py -v

build:  ## Build package distribution
	python -m build

dist: build  ## Create distribution packages

publish:  ## Publish to PyPI (requires authentication)
	python -m twine upload dist/*

dev-setup: install install-dev  ## Setup development environment
	@echo "Development environment setup complete!"
	@echo "Run 'make test-p0' to verify Phase 0 functionality"

# Development workflow shortcuts
check: cli-format cli-lint test-minimal  ## Run format, lint, and minimal tests

quick-check: cli-format test-p0  ## Quick check with format and P0 tests

# Version and status info
status:  ## Show AgentNet phase status
	@PYTHONPATH=. python -c "import agentnet; print('Version:', agentnet.__version__); print('Phase Status:', agentnet.__phase_status__)"

demo:  ## Run Phase 0 demo
	@PYTHONPATH=. python -c "from agentnet import AgentNet, ExampleEngine; engine = ExampleEngine(); agent = AgentNet('Demo', {'logic': 0.8}, engine=engine); result = agent.generate_reasoning_tree('Hello AgentNet!'); print('Demo Result:', result['result']['content'])"

demo-full:  ## Run comprehensive Phase 0 demo
	@PYTHONPATH=. python examples/phase0_demo.py