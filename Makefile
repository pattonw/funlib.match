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
	pytest -v --cov=funlib funlib
	flake8 funlib
