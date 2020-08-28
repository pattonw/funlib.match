default:
	pip install -r requirements.txt
	pip install .

install-dev:
	pip install -r requirements.txt
	pip install -r requirements_dev.txt
	pip install -e .

.PHONY: tests
tests:
	pytest -v --cov=funlib.match funlib/tests -m "not benchmark"
	flake8 funlib
benchmark:
	pytest -v funlib --benchmark-group-by=param:constraint funlib/benchmarks -m "benchmark"
