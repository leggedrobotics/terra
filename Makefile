.PHONY: help coverage linter install mypy test validate frontend-build

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  coverage   to make source code coverage check"
	@echo "  help       to show this help"
	@echo "  install    to make package install"
	@echo "  test       to make tests running"
	@echo "  validate   to make source code validation"

coverage:
	tox -e coverage

linter:
	tox -e linter

mypy:
	tox -e mypy

test:
	tox

validate:
	tox -e pre-commit

all:
	make coverage
	make linter
	make mypy
	make test
	make validate
