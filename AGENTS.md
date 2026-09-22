# scikit-learn — Agent Instructions

## Build & Install

```
pip install --verbose --no-build-isolation --editable . \
  --config-settings editable-verbose=true    # dev install (Meson)
make clean                                   # remove build artifacts if switching numpy <2/>=2
```

- **Build system**: Meson + Cython (not setuptools). Never run `python setup.py`.
- Rebuild is typically automatic on import; `make clean` only needed when ABI changes matter.
- Python >= 3.10, requires NumPy >= 1.22.0, SciPy >= 1.8.0.

## Testing

```
pytest sklearn                               # full suite
pytest sklearn/linear_model/tests/test_linear_model.py  # single file
pytest sklearn/linear_model/tests/test_linear_model.py::test_least_squares  # single test
```

Key environment variables:
- `SKLEARN_SEED` — controls RNG for deterministic test runs.
- `SKLEARN_TESTS_GLOBAL_RANDOM_SEED` — parametrizes tests across seeds `[0,99]` or a range like `"10-20"` (default: `[42]`).
- `SKLEARN_SKIP_NETWORK_TESTS=0` — enables tests that download datasets (newsgroups, LFW, etc.).
- `SKLEARN_RUN_FLOAT32_TESTS=1` — enables float32-specific test paths.
- `SKLEARN_WARNINGS_AS_ERRORS=1` — treats warnings as errors.
- `SCIPY_ARRAY_API=1` — enables Array API dispatch path.

Doctests are skipped on Windows and with numpy < 2 or scipy < 1.14. Doctest directives `ELLIPSIS` and `NORMALIZE_WHITESPACE` are set globally; do **not** write `# doctest: +ELLIPSIS` in code.

## Linting

Run the CI-equivalent lint check:
```
./build_tools/linting.sh                     # ruff check + ruff format + mypy + cython-lint
```

Or manually:
```
ruff check sklearn/                          # lint
ruff format --diff sklearn/                  # format check
mypy sklearn/                                # type check
cython-lint sklearn/                         # Cython lint
```

Pre-commit hooks mirror this (see `.pre-commit-config.yaml`).

**Special lint rules:**
- Use `from sklearn.utils.parallel import delayed` and `Parallel` — **not** `joblib.delayed` or `joblib.Parallel` directly (checked in linting.sh).
- `@deprecated` must come **before** `@property` in decorator order.
- Copyright header is checked via ruff's `CPY001` rule; test files, docs, and tools are exempt.

## Code Structure

- Single package: `sklearn/`. Submodules map directly to ML domains:
  - `sklearn.cluster/`, `sklearn.ensemble/`, `sklearn.linear_model/`, `sklearn.metrics/`, `sklearn.preprocessing/`, `sklearn.svm/`, `sklearn.tree/`, etc.
  - Shared infrastructure: `sklearn/utils/`, `sklearn/base.py`, `sklearn/pipeline.py`, `sklearn/compose/`.
- Cython files: `sklearn/_loss/`, `sklearn/metrics/_dist_metrics.pyx`, `sklearn/linear_model/_sag_fast.pyx`, etc.
- External dependencies vendored in `sklearn/externals/` (excluded from linting).
- Examples live in `examples/`, each subdirectory has a `README.txt`.

## Changelog (towncrier)

- Drop files go in `doc/whats_new/upcoming_changes/<category>/<PR>.rst`.
- Categories: `major-feature`, `feature`, `efficiency`, `enhancement`, `fix`, `api`, `other`.
- Sections: `security`, `changed-models`, `many-modules`, `array-api`, `metadata-routing`, plus per-module dirs.
- PR title must include a changelog fragment, or the `check-changelog` CI workflow blocks the merge.

## CI Quirks

- Azure Pipelines (`azure-pipelines.yml`) is the main CI. GitHub Actions workflows handle docs, wheels, and label automation.
- Commit messages with `[ci skip]` skip all test jobs. `[lint skip]` skips linting.
- Commit messages with `[scipy-dev]` trigger nightly dev-dependency tests.
- Test jobs run with different non-default seeds to catch seed-sensitive bugs.

## Documentation

- Built with Sphinx + pydata-sphinx-theme + sphinx-gallery.
- Generate doc: requires `pip install ".[docs]"` and runs via CircleCI.
- Generated files tracked in `.gitignore`: `doc/_build/`, `doc/auto_examples/`, `doc/api/*.rst`, `doc/modules/generated/`, `doc/whats_new/notes-towncrier.rst`.

## Contributing Guide

Full guide: `doc/developers/contributing.rst` or https://scikit-learn.org/dev/developers/contributing.html
