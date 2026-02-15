##################################################################
# multimin Makefile
##################################################################

.PHONY: help install install-dev show test verify clean build docs push release env gallery paper-pdf

NOTEBOOKS := examples/*.ipynb
GALLERY_TIMEOUT ?= 1200
DOCKER_PLATFORM ?=

RELMODE=release
PYTHON ?= python3
COMMIT_MSG ?= [FIX] Minor changes

help:
	@echo "MultiMin Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install the package"
	@echo "  install-dev  - Install package in development mode with dev dependencies"
	@echo "  show         - Show installed package version"
	@echo "  test         - Run tests with pytest"
	@echo "  verify       - Verify package installation"
	@echo "  clean        - Remove build artifacts and cache files"
	@echo "  build        - Build distribution packages"
	@echo ""
	@echo "  docs         - Build documentation (installs docs requirements)"
	@echo "  gallery     - Run example notebooks to regenerate images in examples/gallery"
	@echo "  paper-pdf    - Build JOSS draft PDF (paper.pdf) via Docker"
	@echo "  push         - Commit (all files) and push current branch"
	@echo "  release      - Release a new version (usage: make release RELMODE=release VERSION=x.y.z)"
	@echo "  env          - Create local dev environment (.multimin) and contrib directory"


show:
	@$(PYTHON) -m pip show multimin 2>/dev/null | awk '/^Version:/{print $$2}' | head -n 1 | grep -E '.' || (echo "multimin is not installed in this Python environment" >&2; exit 1)


env:
	@echo "Creating local development environment..."
	@test -d .multimin || $(PYTHON) -m venv .multimin
	@echo "Installing dependencies..."
	@. .multimin/bin/activate && pip install --upgrade pip
	@. .multimin/bin/activate && pip install -e .
	@if [ -f requirements-dev.txt ]; then . .multimin/bin/activate && pip install -r requirements-dev.txt; fi
	@echo "______________________________________________________________________"
	@echo "Environment setup complete."
	@echo "To activate the environment, run:"
	@echo "source .multimin/bin/activate"



%.md:%.ipynb
	python3 -m nbconvert $^ --to markdown

install:
	$(PYTHON) -m pip install .

install-dev:
	$(PYTHON) -m pip install -e .
	if [ -f requirements.txt ]; then $(PYTHON) -m pip install -r requirements.txt; fi
	if [ -f requirements-dev.txt ]; then $(PYTHON) -m pip install -r requirements-dev.txt; fi

test:
	$(PYTHON) -m pytest src/multimin/tests

verify:
	@chmod +x bin/verify_installation.py
	@$(PYTHON) bin/verify_installation.py

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf tmp/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type d -name "multimin_data" -exec rm -rf {} +
	
build: clean
	$(PYTHON) -m build

gallery:
	@echo "Regenerating gallery images by executing example notebooks..."
	@mkdir -p examples/gallery
	@for nb in $(NOTEBOOKS); do \
		echo "Executing $$nb..."; \
		cd examples && $(PYTHON) -m jupyter nbconvert --execute --to notebook --inplace \
			--ExecutePreprocessor.timeout=$(GALLERY_TIMEOUT) "$$(basename $$nb)" && cd .. || exit 1; \
	done
	@echo "Gallery images updated in examples/gallery/"

docs:
	$(PYTHON) -m pip install -r docs/requirements.txt
	rm -rf docs/_build
	@echo "Preparing documentation..."
	@chmod +x bin/prepare_docs.sh
	@./bin/prepare_docs.sh

	cd docs && $(PYTHON) -m sphinx.cmd.build -M html "." "_build"

paper-pdf:
	@echo "Building paper.pdf using openjournals/inara (JOSS)..."
	@docker run --rm \
		$(if $(DOCKER_PLATFORM),--platform $(DOCKER_PLATFORM),) \
		--volume "$(PWD):/data" \
		--user "$$(id -u):$$(id -g)" \
		--env JOURNAL=joss \
		openjournals/inara
	@echo "Done. Output: paper.pdf"

push:
	@echo "Committing tracked changes (if any)..."
	@if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$$(git status --porcelain)" ]; then \
		git add . && \
		files="$$(git diff --cached --name-only | paste -sd', ' - || true)" && \
		msg="$(COMMIT_MSG)" && \
		if [ "$(origin COMMIT_MSG)" != "command line" ] && [ "$(origin COMMIT_MSG)" != "environment" ]; then \
			if [ -n "$$files" ]; then msg="$$msg [$$files]"; fi; \
		fi && \
		git commit -m "$$msg"; \
	else \
		echo "Working tree is clean (tracked files); nothing to commit."; \
	fi
	@echo "Pushing current branch..."
	@git push -u origin HEAD

# Example: make release RELMODE=release VERSION=0.2.0.2
# Preferred: make release docs gallery VERSION=0.9.5 COMMIT_MSG="[FIX] Consistent option props and properties" && make push COMMIT_MSG="[REL] New version released"
release: clean push
	@test -n "$(VERSION)" || (echo "ERROR: VERSION is required. Example: make release RELMODE=release VERSION=0.2.0" && exit 1)
	@echo "Releasing a new version..."
	@bash bin/release.sh $(RELMODE) $(VERSION)
	@make push COMMIT_MSG="[REL] released version $(VERSION)"

sleep:
	@sleep 3

show:
	echo $(PYTHON)