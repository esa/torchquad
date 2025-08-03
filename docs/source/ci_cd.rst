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

The CI/CD pipeline consists of five main workflows:

1. **Test Suite** (``run_tests.yml``)
   
   **Triggers**: Push to main/develop branches, pull requests, manual dispatch
   
   This is the core testing workflow that runs on every code change:
   
   * **Linting Stage**: Uses flake8 to check code quality and style
   * **Testing Stage**: 
     - Sets up Python 3.9 environment
     - Installs all backend dependencies via micromamba
     - Runs full pytest suite with coverage reporting
     - Posts coverage reports as PR comments
   
   **Key Features**:
   
   * Multi-backend testing (all numerical backends)
   * Coverage tracking with pytest-cov
   * JUnit XML output for CI integration
   * Automated PR comments with test results

2. **Code Formatting** (``autoblack.yml``)
   
   **Triggers**: Pull requests only
   
   Ensures consistent code formatting across the project:
   
   * Uses Black formatter with 100-character line length
   * Python 3.11 environment
   * Checks formatting without modifying files
   * Fails if reformatting is needed

3. **PyPI Deployment** (``deploy_to_pypi.yml``)
   
   **Triggers**: Manual workflow dispatch only
   
   Production deployment to PyPI:
   
   * Python 3.10 environment
   * Builds source distribution and wheel packages
   * Uploads to PyPI using stored authentication token
   * Manual trigger ensures controlled releases

4. **Test PyPI Deployment** (``deploy_to_test_pypi.yml``)
   
   **Triggers**: Manual workflow dispatch, GitHub releases
   
   Test deployment for validation:
   
   * Same process as PyPI deployment
   * Targets Test PyPI for safe testing
   * Used to validate packages before production release

5. **Documentation** (``draft-pdf.yml``)
   
   **Triggers**: Changes to paper directory
   
   Builds academic paper PDF:
   
   * Uses OpenJournals GitHub Action
   * Compiles Markdown to PDF for JOSS submissions
   * Stores generated PDF as workflow artifact

Environment Setup
-----------------

The CI system uses conda/micromamba for dependency management:

.. code-block:: yaml

   # From run_tests.yml
   - name: provision-with-micromamba
     uses: mamba-org/setup-micromamba@v1
     with:
       environment-file: environment_all_backends.yml
       environment-name: torchquad
       cache-downloads: true

Environment Files
~~~~~~~~~~~~~~~~~

* ``environment.yml`` - Basic PyTorch setup for development
* ``environment_all_backends.yml`` - Complete backend support for CI
* ``rtd_environment.yml`` - ReadTheDocs documentation builds

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

Linting with Flake8
~~~~~~~~~~~~~~~~~~~

Two-stage linting process:

1. **Critical Errors**: Check for syntax errors and undefined names
   
   .. code-block:: bash
   
      flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

2. **Full Analysis**: Complete code quality check using project ``.flake8`` configuration
   
   .. code-block:: bash
   
      flake8 . --count --show-source --statistics

Formatting with Black
~~~~~~~~~~~~~~~~~~~~~

Consistent code style enforcement:

.. code-block:: bash

   black --check --line-length 100 .

**Configuration**:

* Line length: 100 characters
* Target: Python 3.11+
* Complies with project style guide

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
   black . --line-length 100
   
   # Check linting
   flake8 . --count --show-source --statistics
   
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
      black . --line-length 100
      git add . && git commit -m "Fix formatting"

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
   * Verify micromamba cache invalidation

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