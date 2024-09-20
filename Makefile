install:
	pip install -r requirements.txt
format:
	black *.py
lint:
	ruff check *.py *.ipynb ./lib/*.py
test:
	python -m pytest -vv --nbval -cov=mylib -cov=main test_main.py main.ipynb
all: install format test lint