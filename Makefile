default:
	conda install -c funkey pylp
	conda install rtree
	pip install -r requirements.txt
	pip install .

install-dev:
	conda install -c funkey pylp
	conda install rtree
	pip install -r requirements_dev.txt
	pip install -e .

.PHONY: tests
tests:
	pytest -v --cov=funlib.match funlib/tests -m "not benchmark"
	flake8 funlib
benchmark:
	pytest -v funlib --benchmark-group-by=param:constraint -m "benchmark"
