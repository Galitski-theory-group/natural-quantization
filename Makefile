.ONESHELL: # Applies to every target in the file!

PYTHON_VERSION ?= $(shell python3 -c "import sys;print('{}.{}'.format(*sys.version_info[:2]))")

# name
.test_repo:
	@echo "PYTHON_VERSION: $(PYTHON_VERSION)"
	python$(PYTHON_VERSION) -m venv .test_repo
	. .test_repo/bin/activate; .test_repo/bin/pip$(PYTHON_VERSION) install --upgrade pip$(PYTHON_VERSION) ; .test_repo/bin/pip$(PYTHON_VERSION) install -e .[dev,test]

test_repo: .test_repo

clean: .test_repo
	rm -rf .test_repo
