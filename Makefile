.PHONY: docs test clean install profile

install:
	pip install -e "."

install-dev:
	pip install -e ".[dev,docs,benchmark]"

install-gpu:
	@echo "Installing JAX with GPU support and development dependencies..."
	./install_gpu.sh

test:
	pytest -xvs tests/

test-coverage:
	pytest --cov=jax_von_mises tests/ --cov-report=xml --cov-report=term

lint:
	flake8 src tests
	black --check src tests
	isort --check-only src tests
	mypy src tests

format:
	black src tests benchmarks docs
	isort src tests benchmarks docs

docs:
	@echo "Building documentation..."
	pip install -e ".[docs]"
	cd docs && sphinx-build -b html . _build/html

docs-serve:
	@echo "Serving documentation on http://localhost:8000"
	cd docs/_build/html && python -m http.server

profile:
	@echo "Running simple profile..."
	python -m benchmarks.simple_profile

profile-gpu:
	@echo "Running GPU profiling (this may take a few minutes)..."
	python -m benchmarks.profile_gpu

benchmark:
	@echo "Running performance benchmarks..."
	python -m benchmarks.performance_benchmark

clean:
	rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ docs/_build/ profile_results/

dist:
	python -m build

publish-test:
	python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

publish:
	python -m twine upload dist/* 