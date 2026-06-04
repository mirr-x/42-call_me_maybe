install:
	pip install --upgrade pip
	pip install mypy
	pip install flake8
	pip install pydantic
	pip install pytest


run:
	

debug:
	

clean:
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.mypy_cache' -exec rm -rf {} +

lint:
	flake8 . --exclude .venv
	mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs --exclude .venv

