Continuous Integration and Deployment
=====================================

This document describes the continuous integration (CI) and continuous deployment (CD) setup for torchquad, which ensures code quality, testing, and automated releases.

Overview
--------

Torchquad uses GitHub Actions for CI/CD with the following key objectives:

* **Automated Testing**: Run comprehensive test suites on every code change
* **Code Quality**: Enforce consistent formatting and linting standards  
* **Multi-Backend Support**: Test across PyTorch, JAX, TensorFlow, and NumPy
* **Automated Deployment**: Streamlined releases to PyPI and Test PyPI
* **Documentation**: Automated paper builds for JOSS submissions

GitHub Actions Workflows
-------------------------

The CI/CD pipeline consists of six main workflows:

1. **Test Suite** (``run_tests.yml``)

   **Triggers**: Push to main/develop branches, pull requests, manual dispatch

   This is the core testing workflow that runs on every code change. All jobs
   install dependencies with `uv <https://docs.astral.sh/uv/>`_ (CPU wheels),
   which is much faster than conda in CI:

   * **Lint**: `ruff <https://docs.astral.sh/ruff/>`_ formatting
     (``ruff format --check``) and code quality (``ruff check``), plus
     `pydoclint <https://github.com/jsh9/pydoclint>`_ Google-style docstring
     checks.
   * **test**: all four backends installed across a Python 3.10/3.11/3.12
     matrix (``fail-fast: false``). Skipped tests are errors
     (``--error-for-skips``), so a silently-missing backend fails the build.
     The 3.12 leg enforces the coverage floor (``--cov-fail-under``) and posts
     the PR coverage comment.
   * **test-single-backend**: pins JAX or TensorFlow in isolation (with a CPU
     torch baseline for collection) so a breaking release of one backend is
     caught on its own.
   * **wheel-smoke**: builds the wheel, installs it clean, and runs the
     built-in deployment self-test — catches packaging breakage before release.
   * **docs-build**: ``sphinx-build -W`` (warnings as errors) so documentation
     rot fails the PR instead of shipping silently to Read the Docs.

   **Key Features**:

   * Multi-backend testing across a Python version matrix
   * Coverage tracking with pytest-cov and an enforced floor
   * JUnit XML output for CI integration
   * Automated PR comments with test results

2. **Dead Code** (``dead_code.yml``)

   **Triggers**: Push to main/develop branches, pull requests, manual dispatch

   Detects unused code with `vulture <https://github.com/jendrikseipp/vulture>`_
   in two tiers:

   * A blocking tier at 100% confidence (near-certain dead code fails the build)
   * An advisory tier at 60% confidence (reported but non-blocking)

   Genuine implicit uses go in ``.vulture_whitelist.py``, but deleting dead code
   is preferred over whitelisting it.

3. **PyPI Deployment** (``deploy_to_pypi.yml``)
   
   **Triggers**: Manual workflow dispatch only
   
   Production deployment to PyPI:
   
   * Python 3.11 environment
   * Builds source distribution and wheel packages
   * Uploads to PyPI using stored authentication token
   * Manual trigger ensures controlled releases

4. **Test PyPI Deployment** (``deploy_to_test_pypi.yml``)
   
   **Triggers**: Manual workflow dispatch only
   
   Test deployment for validation:
   
   * Same process as PyPI deployment
   * Targets Test PyPI for safe testing
   * Used to validate packages before production release

5. **Release Testing** (``release_testing.yml``)

   **Triggers**: Any push to a ``release-*`` or ``release/**`` branch, manual
   workflow dispatch, and GitHub releases. A hotfix branch named otherwise has to
   be dispatched by hand.

   The slower end-to-end suite in ``release_testing/``, run against the *latest
   released* backends rather than the versions pinned in CI, so a release cannot
   silently break on a new torch/JAX/TensorFlow:

   * Python 3.10/3.11/3.12 matrix
   * Accuracy, seeded determinism, autodiff and JIT-vs-eager parity checks

6. **Documentation** (``draft-pdf.yml``)
   
   **Triggers**: Changes to paper directory
   
   Builds academic paper PDF:
   
   * Uses OpenJournals GitHub Action
   * Compiles Markdown to PDF for JOSS submissions
   * Stores generated PDF as workflow artifact

Environment Setup
-----------------

CI installs dependencies with `uv <https://docs.astral.sh/uv/>`_ (CPU wheels),
which is the primary dev/CI toolchain:

.. code-block:: yaml

   # From run_tests.yml
   - name: Set up uv
     uses: astral-sh/setup-uv@v5
     with:
       enable-cache: true
   - name: Install torchquad and all backends (CPU)
     run: |
       uv venv --python 3.12
       uv pip install torch --index-url https://download.pytorch.org/whl/cpu
       uv pip install "jax[cpu]" tensorflow numpy
       uv pip install -e ".[dev]"

Dependency management paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``uv`` + ``uv.lock`` - primary reproducible dev/CI toolchain
* ``environment.yml`` - basic conda setup for development
* ``environment_all_backends.yml`` - complete backend support (conda path)
* ``pixi.toml`` - experimental per-backend environments for pixi users
* ``rtd_environment.yml`` - ReadTheDocs documentation builds

Locally, ``uv sync --extra all`` (or ``uv pip install -e ".[dev]"`` plus the
backend of your choice) reproduces the dev environment.

Test Execution
--------------

The test suite runs with comprehensive coverage:

.. code-block:: bash

   cd tests/
   pytest -ra --error-for-skips \\
          --junitxml=pytest.xml \\
          --cov-report=term-missing:skip-covered \\
          --cov=../torchquad . | tee pytest-coverage.txt

**Test Parameters**:

* ``-ra`` - Show summary for all test outcomes
* ``--error-for-skips`` - Treat skipped tests as errors (fail CI)
* ``--junitxml`` - Generate XML report for CI integration
* ``--cov`` - Generate coverage report for the torchquad package

Code Quality Standards
----------------------

Linting and Formatting with Ruff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`ruff <https://docs.astral.sh/ruff/>`_ replaces the previous flake8 + black
setup with a single, much faster tool. Configuration lives in the ``[tool.ruff]``
section of ``pyproject.toml`` (line length 100).

1. **Formatting**: Check (or apply) the code style

   .. code-block:: bash

      ruff format --check .   # check only (CI)
      ruff format .           # apply

2. **Critical Errors**: Syntax errors and undefined names (the severe gate)

   .. code-block:: bash

      ruff check --select=E9,F63,F7,F82 .

3. **Full Analysis**: Complete lint check

   .. code-block:: bash

      ruff check .

Docstrings with Pydoclint
~~~~~~~~~~~~~~~~~~~~~~~~~~~

`pydoclint <https://github.com/jsh9/pydoclint>`_ enforces Google-style docstring
completeness and consistency (missing ``Args``/``Returns``/``Raises``) on the
``torchquad`` package. Configuration lives in ``[tool.pydoclint]`` in
``pyproject.toml``.

.. code-block:: bash

   pydoclint torchquad/

Dead Code with Vulture
~~~~~~~~~~~~~~~~~~~~~~~~

`vulture <https://github.com/jendrikseipp/vulture>`_ flags unused code. The
blocking CI tier runs at 100% confidence; a 60% advisory tier is non-blocking.

.. code-block:: bash

   vulture torchquad .vulture_whitelist.py --min-confidence 100

Pre-commit Hooks
~~~~~~~~~~~~~~~~~

All of the above are wired into `pre-commit <https://pre-commit.com/>`_ via
``.pre-commit-config.yaml``. Install them once so issues are caught before pushing:

.. code-block:: bash

   pip install pre-commit
   pre-commit install
   pre-commit run --all-files   # run against the whole repo

Coverage Reporting
------------------

The CI system provides detailed coverage analysis:

* **PR Comments**: Automated coverage reports on pull requests
* **Trend Tracking**: Coverage change detection
* **Missing Lines**: Identification of untested code
* **Badge Integration**: Coverage badges for README

**Coverage Requirements**:

* New features must include comprehensive tests
* Significant coverage decreases block PR merges
* Target: >90% coverage for new code

Local Development
-----------------

Before pushing changes, run these checks locally:

.. code-block:: bash

   # Format code
   ruff format .

   # Check linting and docstrings
   ruff check .
   pydoclint torchquad/

   # Run tests
   cd tests/
   pytest

   # Run with coverage
   pytest --cov=../torchquad

Environment Setup
~~~~~~~~~~~~~~~~~

For local development:

.. code-block:: bash

   # Create environment
   conda env create -f environment_all_backends.yml
   conda activate torchquad
   
   # Install in development mode
   pip install -e .

Backend Testing
---------------

Multi-Backend Strategy
~~~~~~~~~~~~~~~~~~~~~~

Tests run across all supported numerical backends:

* **NumPy**: Reference implementation and baseline testing
* **PyTorch**: GPU acceleration and automatic differentiation
* **JAX**: JIT compilation and XLA optimization
* **TensorFlow**: Graph execution and TPU support

**Backend-Specific Considerations**:

* Some tests are backend-specific and use appropriate skip decorators
* GPU tests run automatically when CUDA is available
* Complex number support varies by backend
* Performance characteristics differ between backends

Release Process
---------------

PyPI Deployment
~~~~~~~~~~~~~~~

Production releases follow this process:

1. **Code Review**: All changes go through PR review
2. **Testing**: Full test suite must pass
3. **Version Update**: Update version in ``pyproject.toml``
4. **Test Deployment**: Deploy to Test PyPI first
5. **Validation**: Test installation from Test PyPI
6. **Production**: Manual trigger of PyPI deployment workflow

**Required Secrets**:

* ``PYPI_TOKEN`` - PyPI API token for package uploads
* ``TEST_PYPI_TOKEN`` - Test PyPI API token

Security Considerations
-----------------------

* **Token Management**: API tokens stored as GitHub secrets
* **Manual Triggers**: Production deployments require manual approval
* **Branch Protection**: Main branch protected with required status checks
* **Dependency Scanning**: Automated security updates via Dependabot

Troubleshooting
---------------

Common CI Failures
~~~~~~~~~~~~~~~~~~

1. **Formatting Issues**:

   .. code-block:: bash

      # Fix locally
      ruff format .
      git add . && git commit -m "style: fix formatting"

2. **Import Errors**:
   
   * Check dependency versions in environment files
   * Verify relative imports after package structure changes
   * Ensure test files are properly isolated

3. **Backend-Specific Failures**:
   
   * Check if backend is properly installed in CI environment
   * Verify skip decorators for unavailable backends
   * Review backend-specific test logic

4. **Coverage Decreases**:
   
   * Add tests for new functionality
   * Check test discovery (files must match ``*_test.py`` or ``test_*.py``)
   * Verify coverage configuration in ``pyproject.toml``

5. **Environment Issues**:
   
   * Update ``environment_all_backends.yml`` for new dependencies
   * Check for version conflicts between backends
   * Verify the ``uv`` cache is invalidated when ``uv.lock`` changes

Building Documentation Locally
------------------------------

To build the Sphinx documentation locally:

.. code-block:: bash

   # Navigate to docs directory
   cd docs
   
   # Build HTML documentation
   make html
   
   # On Windows, you can also use:
   make.bat html
   
   # Clean build directory
   make clean
   
   # View all available targets
   make help

The built documentation will be available in ``docs/_build/html/``. Open ``docs/_build/html/index.html`` in your browser to view the documentation.

**Note**: Make sure you have Sphinx and all documentation dependencies installed:

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme

Getting Help
------------

For CI/CD issues:

1. Check the `GitHub Actions <https://github.com/esa/torchquad/actions>`_ page for detailed logs
2. Review similar successful runs for comparison
3. Check environment file consistency
4. Verify all required secrets are configured
5. Open an issue with CI logs if problems persist

The CI/CD system is designed to catch issues early and ensure high code quality. 
When in doubt, run the same commands locally that CI runs to debug issues quickly.