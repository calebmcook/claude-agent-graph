# Testing Setup and Configuration Guide

This guide explains how to set up and use the hybrid testing infrastructure for the Agent Graph Flask application.

## Quick Start

### 1. Install Dependencies

```bash
# Install the package with development dependencies
pip install -e ".[dev]"

# Install Playwright browsers (for E2E tests)
playwright install
```

### 2. Set Up Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# Verify installation
pre-commit run --all-files
```

### 3. Run Tests

```bash
# Run only unit tests (fastest)
pytest tests/test_flask_api.py -v

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/test_flask_api.py --cov=app --cov-report=html
```

## Detailed Setup Guide

### Environment Variables

Create a `.env` file for local testing:

```bash
# Flask configuration
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_APP=app.py

# For manual testing with real Claude API
ANTHROPIC_API_KEY=your-api-key-here

# For automated testing (optional)
TESTING_MODE=true
```

Load environment variables:
```bash
source .env  # On Linux/Mac
# or
set -a && source .env && set +a
```

### Pre-commit Hooks Configuration

The `.pre-commit-config.yaml` file defines hooks that run automatically before commits:

#### What Hooks Run

1. **Black** - Code formatting
   - Ensures consistent code style
   - Auto-fixes formatting issues

2. **Ruff** - Linting
   - Checks for code quality issues
   - Auto-fixes many issues with `--fix`

3. **Type Checking** - mypy
   - Validates type annotations
   - Optional in main (continue-on-error)

4. **General Checks**
   - Trailing whitespace
   - End of file newlines
   - YAML formatting
   - JSON validation
   - Merge conflict markers

5. **Unit Tests** - pytest
   - Runs Flask API tests
   - Requires all tests to pass before commit

#### Skipping Hooks

If you need to skip hooks temporarily:

```bash
# Skip all hooks
git commit --no-verify

# Skip pre-commit via environment variable
SKIP=pytest-unit-tests git commit -m "message"

# Only skip specific hooks
pre-commit run --hook-stage commit --skip=pytest-unit-tests
```

#### Updating Hooks

```bash
# Update all pre-commit hooks to latest versions
pre-commit autoupdate

# Update specific hook
pre-commit autoupdate --repo https://github.com/psf/black
```

## Testing Guide

### Unit Tests (Tier 1 - Fast)

Unit tests verify API endpoints and business logic with mocked Claude API responses.

#### Running Unit Tests

```bash
# Run all unit tests
pytest tests/test_flask_api.py -v

# Run specific test class
pytest tests/test_flask_api.py::TestInitializeEndpoint -v

# Run specific test
pytest tests/test_flask_api.py::TestInitializeEndpoint::test_initialize_with_valid_problem -v

# Run tests matching pattern
pytest tests/test_flask_api.py -k "test_initialize" -v

# Run with output
pytest tests/test_flask_api.py -v -s

# Run quietly
pytest tests/test_flask_api.py -q
```

#### Coverage Analysis

```bash
# Generate coverage report
pytest tests/test_flask_api.py --cov=app --cov-report=html

# View HTML report
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
```

#### Debugging Failed Tests

```bash
# Show print statements
pytest tests/test_flask_api.py -s

# Show local variables on failure
pytest tests/test_flask_api.py -l

# Drop into debugger on failure
pytest tests/test_flask_api.py --pdb

# Show slowest tests
pytest tests/test_flask_api.py --durations=10
```

### E2E Tests (Tier 2 - Medium Speed)

E2E tests use Playwright to test complete workflows in a real browser.

#### Prerequisites

```bash
# Install Playwright browsers
playwright install

# Or install specific browser
playwright install chromium
```

#### Running E2E Tests

```bash
# Run all E2E tests
pytest tests/test_flask_e2e_playwright.py -v

# Run specific test class
pytest tests/test_flask_e2e_playwright.py::TestInitializationWorkflow -v

# Run in headed mode (see the browser)
pytest tests/test_flask_e2e_playwright.py --headed -v

# Run with specific browser
pytest tests/test_flask_e2e_playwright.py --browser chromium
pytest tests/test_flask_e2e_playwright.py --browser firefox
pytest tests/test_flask_e2e_playwright.py --browser webkit

# Run in debug mode
pytest tests/test_flask_e2e_playwright.py --debug

# Run with slow-motion (see what's happening)
pytest tests/test_flask_e2e_playwright.py --slowmo 1000
```

#### E2E Test Requirements

The Flask server must be running for E2E tests:

```bash
# In one terminal, start the Flask server
python3 app.py

# In another terminal, run E2E tests
pytest tests/test_flask_e2e_playwright.py -v --headed
```

#### Playwright Inspector

Debug E2E tests with Playwright Inspector:

```bash
# Run with Playwright Inspector
PWDEBUG=1 pytest tests/test_flask_e2e_playwright.py::TestInitializationWorkflow::test_navigate_to_app -v
```

### Manual Testing (Tier 3 - Real API)

Test with real Claude API responses for exploration and debugging.

#### Starting the Development Server

```bash
# Start with hot reload
FLASK_ENV=development python3 app.py

# Server starts at http://localhost:5001
```

#### Features Available

- **Real Claude API**: Actual responses from Claude
- **Hot Reload**: Auto-restart on file changes
- **Flask Debugger**: Built-in debugging interface
- **Debug Mode**: Print statements visible in console

#### Using the Web Interface

1. Open http://localhost:5001 in your browser
2. Enter a problem statement
3. Click "Initialize" to create graph
4. Click "Supervisor Thinks" to analyze
5. Click "Delegate" to assign tasks
6. Get agent responses and chat with supervisor
7. Watch graph visualization update in real-time

#### Debugging Tips

Check Flask console output for:
- Request/response logs
- Claude API calls and responses
- Error messages with full stack traces

Use browser DevTools (F12):
- **Console**: View JavaScript errors
- **Network**: Inspect API requests/responses
- **Elements**: Debug HTML structure

## CI/CD Pipeline

The `.github/workflows/tests.yml` file defines automated testing on:
- Push to main, develop, or feature branches
- Pull requests to main/develop

### Pipeline Stages

#### 1. Unit Tests (All Python Versions)
Runs on Python 3.10, 3.11, 3.12:
- Flask API tests
- Coverage reporting
- Upload to Codecov

#### 2. Integration Tests
- Optional integration tests
- Marked with `@pytest.mark.integration`

#### 3. E2E Tests
- Playwright tests
- Starts Flask server
- Tests on Chromium browser

#### 4. Code Quality
- Black formatting check
- Ruff linting
- mypy type checking (optional)

#### 5. Test Summary
- Aggregates results
- Posts summary to PR

### Checking CI Results

1. **In GitHub UI**
   - View status next to commit
   - Click "Details" to see logs
   - Check PR "Checks" tab

2. **Via Command Line**
   ```bash
   # View workflow runs
   gh run list

   # View specific run
   gh run view <run-id>
   ```

### Handling Failures

**Unit Test Failures**
- Must be fixed before merge
- Run locally: `pytest tests/test_flask_api.py -v`
- Check logs in GitHub Actions

**Code Quality Failures**
- Auto-fix with: `black src/ tests/ && ruff check --fix`
- Commit fixes and push

**E2E Test Failures**
- Currently allowed to fail (continue-on-error: true)
- Run locally with `--headed` to debug: `pytest tests/test_flask_e2e_playwright.py --headed`

## Development Workflow

### Recommended Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make changes and write tests**
   ```bash
   # Edit code
   vim src/mymodule.py

   # Write tests
   vim tests/test_mymodule.py

   # Run tests locally
   pytest tests/test_mymodule.py -v
   ```

3. **Pre-commit hooks run automatically**
   ```bash
   git add .
   git commit -m "Add new feature"

   # Pre-commit hooks:
   # 1. Black formats code
   # 2. Ruff lints and fixes
   # 3. mypy checks types
   # 4. pytest runs unit tests
   ```

4. **Push to GitHub**
   ```bash
   git push origin feature/my-feature
   ```

5. **CI/CD Pipeline runs**
   - GitHub Actions runs all checks
   - Results appear in PR

6. **Fix issues if needed**
   ```bash
   # If tests fail in CI
   pytest tests/test_flask_api.py -v
   # Fix issues locally
   git add . && git commit -m "Fix tests"
   git push
   ```

7. **Merge when ready**
   - All checks pass
   - Code review approved
   - Merge to main

## Troubleshooting

### Pre-commit Hooks Not Running

```bash
# Verify installation
ls -la .git/hooks/

# Reinstall hooks
pre-commit install --install-hooks

# Run hooks manually
pre-commit run --all-files
```

### Unit Tests Fail Locally but Pass in CI

```bash
# Make sure you're testing the right version
pip install -e ".[dev]"  # Reinstall in editable mode

# Clear cache
rm -rf .pytest_cache
pytest tests/test_flask_api.py -v
```

### E2E Tests Timeout

```bash
# Increase timeout
pytest tests/test_flask_e2e_playwright.py --timeout=60 -v

# Or in test code
await page.wait_for_timeout(10000)  # 10 seconds
```

### Flask Server Won't Start

```bash
# Check if port 5001 is in use
lsof -i :5001

# Kill existing process
kill -9 <PID>

# Try different port
FLASK_PORT=5002 python3 app.py
```

### Playwright Browsers Not Found

```bash
# Install all browsers
playwright install

# Or install specific browser
playwright install chromium

# Verify installation
playwright install --with-deps
```

### Mock Not Working in Tests

```bash
# Make sure to patch at import location
@patch("app.ClaudeSDKClient")  # Correct
# NOT: @patch("claude_agent_sdk.ClaudeSDKClient")  # Wrong

def test_something(mock_client):
    pass
```

## Advanced Configuration

### Pytest Configuration

Edit `pyproject.toml` `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",
]
asyncio_mode = "auto"
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration",
]
```

### Playwright Configuration

Create `pytest.ini` or `pyproject.toml`:

```ini
[pytest]
asyncio_mode = auto
```

### Custom Fixtures

Add custom fixtures in `tests/conftest.py`:

```python
@pytest.fixture
def my_custom_fixture():
    """Custom fixture description"""
    setup_code()
    yield value
    teardown_code()
```

## Performance Optimization

### Parallel Test Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest tests/test_flask_api.py -n auto
```

### Test Caching

```bash
# Cache test results
pytest tests/test_flask_api.py --cache-show
```

## Continuous Learning

- **pytest**: https://docs.pytest.org/
- **Playwright**: https://playwright.dev/python/
- **Flask Testing**: https://flask.palletsprojects.com/testing/
- **Pre-commit**: https://pre-commit.com/

## Getting Help

- Check test output for errors: `pytest -v -s`
- Use debugger: `pytest --pdb`
- Run with verbose output: `pytest -vv`
- Check logs: Flask logs in terminal, GitHub Actions logs online
