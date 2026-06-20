install:
	uv sync


run:
	uv run python -m call_me_maybe

debug:
	uv run python -m pdb -m call_me_maybe.main

clean:
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.mypy_cache' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +

lint:
	flake8 . --exclude .venv
	mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs --exclude .venv

