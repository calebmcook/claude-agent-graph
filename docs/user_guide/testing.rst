========
Testing
========

The claude-agent-graph project uses a **comprehensive three-tier hybrid testing strategy** to ensure code quality, reliability, and maintainability across development stages.

Overview
========

The testing approach combines:

1. **Unit & Integration Tests** - Fast feedback for development
2. **End-to-End Tests** - Complete workflow validation
3. **Manual Testing** - Real-world exploration and debugging

Testing Pyramid
===============

.. code-block:: text

        ┌─────────────────────────────────┐
        │                                 │
        │   Manual Testing with          │
        │   Flask Development Server     │
        │   (Real Claude API)            │
        │   ~1 per development session   │
        │                                 │
        ├─────────────────────────────────┤
        │                                 │
        │   Playwright E2E Tests         │
        │   (Complete workflows)         │
        │   ~20-30 tests                 │
        │   ~15-30 seconds per run       │
        │                                 │
        ├─────────────────────────────────┤
        │                                 │
        │   Pytest Unit & Integration    │
        │   (API endpoints, mocked APIs) │
        │   ~50-100 tests                │
        │   ~1-2 seconds per test        │
        │                                 │
        └─────────────────────────────────┘

Tier 1: Unit & Integration Tests
=================================

Fast feedback tests using pytest with mocked Claude API responses.

**Purpose:** Validate API contracts, error handling, and business logic

**Duration:** <2 seconds per test

**Running Tests:**

.. code-block:: bash

    # Run Flask API unit tests
    pytest tests/test_flask_api.py -v

    # Run with coverage report
    pytest tests/test_flask_api.py --cov=app --cov-report=html

    # Run all unit tests
    pytest tests/ -k "not e2e" -v

    # Run specific test class
    pytest tests/test_flask_api.py::TestInitializeEndpoint -v

    # Run with output
    pytest tests/test_flask_api.py -v -s

**Test Files:**

- ``tests/test_flask_api.py`` - Flask API endpoints (26 tests)
- ``tests/test_integration.py`` - Integration scenarios (17+ tests)
- ``tests/conftest.py`` - Pytest fixtures and mocking setup

**Coverage:**

- Initialize endpoint
- Supervisor analysis endpoint
- Task delegation endpoint
- Agent response endpoint
- Supervisor chat endpoint
- Graph state endpoint
- Error handling
- CORS headers

**Fixtures:**

The ``conftest.py`` provides comprehensive fixtures:

- ``app`` - Flask test application
- ``client`` - Flask test client
- ``mock_supervisor_analysis`` - Mock Claude response
- ``mock_delegation_response`` - Mock task delegation
- ``patch_claude_sdk`` - Mocked ClaudeSDKClient
- ``patch_agent_graph`` - Mocked AgentGraph

Tier 2: End-to-End Tests
========================

Browser-based tests using Playwright for complete workflow validation.

**Purpose:** Test complete user journeys and UI interactions

**Duration:** 15-30 seconds per test

**Installation:**

.. code-block:: bash

    # Install Playwright browsers (one-time)
    playwright install chromium

    # Or install all browsers
    playwright install

**Running Tests:**

.. code-block:: bash

    # Start Flask server
    python3 app.py &

    # Run E2E tests in headless mode
    pytest tests/test_flask_e2e_playwright.py -v

    # Run with visible browser (headed mode)
    pytest tests/test_flask_e2e_playwright.py --headed -v

    # Run specific test class
    pytest tests/test_flask_e2e_playwright.py::TestInitializationWorkflow -v

    # Run with debug mode
    PWDEBUG=1 pytest tests/test_flask_e2e_playwright.py -v

**Test Coverage:**

- Problem initialization workflow
- Supervisor analysis and agent creation
- Task delegation flow
- Agent response handling
- Supervisor chat interactions
- Graph visualization updates
- Error handling in UI
- Loading states and responsiveness

**Test File:** ``tests/test_flask_e2e_playwright.py`` (40+ tests)

Tier 3: Manual Testing
======================

Interactive testing with the Flask development server using the real Claude API.

**Purpose:** Exploration, debugging, acceptance testing with real responses

**Starting the Server:**

.. code-block:: bash

    # With hot reload enabled
    FLASK_ENV=development python3 app.py

    # Server runs on http://localhost:5001

**Using the Web Interface:**

1. Open http://localhost:5001 in your browser
2. Enter a problem statement
3. Click "Initialize" to create the graph
4. Click "Supervisor Thinks" to analyze the problem
5. Click "Delegate" to assign tasks to agents
6. Get individual agent responses
7. Chat with the supervisor

**Features:**

- Real Claude API responses
- Automatic file reload on changes
- Flask debug toolbar for debugging
- Full logging output in console
- Interactive graph visualization

**Debugging Tips:**

.. code-block:: bash

    # Check Flask logs for errors
    tail -f /tmp/flask.log

    # Use browser DevTools (F12)
    # - Console: JavaScript errors
    # - Network: API request/response
    # - Elements: HTML inspection

Pre-commit Hooks
================

Automatically run tests and checks before commits.

**Installation:**

.. code-block:: bash

    pip install pre-commit
    pre-commit install

**Configured Hooks:**

- Black - Code formatting
- Ruff - Linting and code quality
- mypy - Type checking
- General checks (whitespace, YAML, JSON)
- **pytest** - Unit test execution (must pass)

**Running Hooks:**

.. code-block:: bash

    # Run all hooks on modified files
    pre-commit run

    # Run all hooks on all files
    pre-commit run --all-files

    # Skip hooks (not recommended)
    git commit --no-verify

    # Update hooks to latest versions
    pre-commit autoupdate

CI/CD Pipeline
==============

Automated testing runs on GitHub Actions for every push and pull request.

**Configuration:** ``.github/workflows/tests.yml``

**Test Matrix:**

- Python versions: 3.10, 3.11, 3.12
- Operating systems: Ubuntu latest, macOS latest
- Jobs: unit tests, integration tests, code quality checks

**Pipeline Stages:**

1. **Unit Tests** - Runs Flask API tests with coverage reporting
2. **Integration Tests** - Runs integration test suite
3. **E2E Tests** - Runs Playwright E2E tests (allowed to fail initially)
4. **Code Quality** - Checks Black, Ruff, mypy
5. **Test Summary** - Aggregates results and posts to PR

**Coverage Reporting:**

- Results uploaded to Codecov
- Coverage badges displayed on PR
- Target: >80% coverage

Common Commands
===============

**Quick Testing (5-10 seconds)**

.. code-block:: bash

    pytest tests/test_flask_api.py -q

**Full Testing (30-60 seconds)**

.. code-block:: bash

    pytest tests/test_flask_api.py tests/test_flask_e2e_playwright.py -v

**With Coverage Report**

.. code-block:: bash

    pytest tests/test_flask_api.py --cov=app --cov-report=html
    open htmlcov/index.html

**E2E with Headed Browser**

.. code-block:: bash

    python3 app.py &
    pytest tests/test_flask_e2e_playwright.py --headed -v

Troubleshooting
===============

**Tests Timeout**

.. code-block:: bash

    # Increase timeout
    pytest tests/ --timeout=60

**Flask Server Not Running for E2E**

.. code-block:: bash

    # Verify server is running
    curl http://localhost:5001

    # Start manually
    python3 app.py

**Import Errors**

.. code-block:: bash

    # Reinstall in editable mode
    pip install -e ".[dev]"

    # Clear cache
    rm -rf .pytest_cache __pycache__

**Playwright Browsers Not Found**

.. code-block:: bash

    # Install browsers
    playwright install chromium

    # Install all browsers
    playwright install

**Mock Not Working**

Make sure to patch at the import location:

.. code-block:: python

    @patch("app.ClaudeSDKClient")  # Correct
    # NOT: @patch("claude_agent_sdk.ClaudeSDKClient")  # Wrong
    def test_something(mock_client):
        pass

Best Practices
==============

**Unit Tests**

- Use fixtures for setup
- Mock external APIs
- Test one thing per test
- Use descriptive names
- Verify side effects

**E2E Tests**

- Wait for elements explicitly
- Use meaningful selectors
- Test user flows, not implementation
- Screenshot on failure
- Run headless by default

**Manual Testing**

- Test real workflows
- Check edge cases
- Verify error messages
- Test with different input
- Performance monitoring

Performance Targets
===================

- Unit tests: <2 seconds each
- E2E tests: 15-30 seconds each
- Full test suite: <5 minutes
- Code coverage: >80%
- CI/CD pass rate: 100%

Next Steps
==========

1. **Increase Coverage** - Add tests for uncovered code paths
2. **Performance Tests** - Monitor response times and throughput
3. **Load Tests** - Test with multiple concurrent users
4. **Visual Regression** - Detect unintended UI changes
5. **Integration** - Test with real backend services

Resources
=========

- `Pytest Documentation <https://docs.pytest.org/>`_
- `Playwright Documentation <https://playwright.dev/python/>`_
- `Flask Testing Guide <https://flask.palletsprojects.com/testing/>`_
- `Pre-commit Framework <https://pre-commit.com/>`_

See Also
========

- :doc:`/HYBRID_TESTING_STRATEGY` - Comprehensive testing strategy
- :doc:`/TESTING_SETUP` - Detailed setup and configuration
- `GitHub Actions Workflow <../.github/workflows/tests.yml>`_
