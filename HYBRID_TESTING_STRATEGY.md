# Hybrid Testing Strategy for Flask Application

## Overview

This document describes the comprehensive three-tier testing strategy for the Agent Graph Flask web application. The strategy combines **unit/integration tests**, **end-to-end browser tests**, and **local development with hot reload** to enable rapid iteration and maintain high code quality.

## Testing Pyramid

```
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
```

## Tier 1: Unit & Integration Tests (Fast, Mocked)

### Purpose
- **Fast feedback loop**: 1-2 seconds per test
- **Isolated testing**: No external dependencies
- **API contract validation**: Ensure endpoints work as expected
- **Error handling**: Test edge cases and failures

### Technology Stack
- **Framework**: pytest
- **Async support**: pytest-asyncio
- **Mocking**: unittest.mock
- **Coverage**: pytest-cov

### Test Coverage

#### Flask API Endpoints (`tests/test_flask_api.py`)
- `/api/initialize` - Graph initialization
- `/api/supervisor-think` - Supervisor analysis
- `/api/delegate` - Task delegation
- `/api/agent-response` - Agent responses
- `/api/supervisor-chat` - Supervisor chat
- `/api/graph-state` - Graph state retrieval
- `/` - Index page serving

#### Test Categories

**1. Happy Path Tests**
```python
def test_initialize_with_valid_problem():
    """Test successful initialization"""
    response = client.post("/api/initialize",
                          json={"problem": "..."})
    assert response.status_code == 200
```

**2. Error Handling Tests**
```python
def test_initialize_without_problem():
    """Test error when problem missing"""
    response = client.post("/api/initialize",
                          json={"problem": ""})
    assert response.status_code == 400
```

**3. Contract Tests**
```python
def test_api_response_structure():
    """Verify response JSON schema"""
    response = client.post("/api/initialize", ...)
    data = json.loads(response.data)
    assert "nodes" in data["graph"]
    assert "links" in data["graph"]
```

### Running Unit Tests

```bash
# Run all unit tests
pytest tests/test_flask_api.py

# Run with coverage
pytest tests/test_flask_api.py --cov=app

# Run specific test
pytest tests/test_flask_api.py::TestInitializeEndpoint::test_initialize_with_valid_problem

# Run with verbose output
pytest tests/test_flask_api.py -v
```

### Fixtures & Mocking

The `tests/conftest.py` provides reusable fixtures:

```python
@pytest.fixture
def client():
    """Flask test client"""
    return app.test_client()

@pytest.fixture
def mock_supervisor_analysis():
    """Mock Claude API response"""
    return {"analysis": "...", "agents_needed": [...]}

@pytest.fixture
def patch_claude_sdk():
    """Mock ClaudeSDKClient"""
    with patch("app.ClaudeSDKClient") as mock:
        yield mock
```

### Best Practices

1. **Use fixtures for setup**: Avoid repetitive test setup
2. **Mock external APIs**: Don't call real Claude API in tests
3. **Test one thing**: Each test should verify single behavior
4. **Descriptive names**: `test_initialize_without_problem` vs `test_init`
5. **Verify side effects**: Check database state, graph modifications, etc.

## Tier 2: Playwright E2E Tests (Complete Workflows)

### Purpose
- **Full workflow validation**: Test complete user journeys
- **Browser compatibility**: Ensure UI works across browsers
- **Integration testing**: Test real interactions between components
- **Visual regression**: Detect unintended UI changes

### Technology Stack
- **Framework**: pytest-playwright
- **Browsers**: Chromium, Firefox, WebKit (configurable)
- **Assertions**: Async-friendly locators

### Test Coverage

#### Complete Workflows (`tests/test_flask_e2e_playwright.py`)

**1. Initialization Workflow**
- Navigate to app
- Enter problem statement
- Click initialize button
- Verify graph is created

```python
async def test_complete_workflow_initialization_to_analysis():
    await page.goto(base_url)
    await page.fill("textarea[id='problemInput']", "...")
    await page.click("button:has-text('Initialize')")
    await expect(svg).to_be_visible()
```

**2. Analysis Workflow**
- Initialize graph
- Click "Supervisor Thinks"
- Verify agents are created
- Check graph visualization updates

**3. Delegation Workflow**
- Complete analysis
- Click "Delegate"
- Verify task delegations appear
- Check agent list updates

**4. Agent Response Workflow**
- Get delegations
- Click agent response buttons
- Verify responses appear
- Check content quality

**5. Chat Workflow**
- Initialize
- Send message to supervisor
- Verify response appears
- Check context maintenance

**6. Graph Visualization**
- Verify SVG renders
- Check nodes appear
- Verify links connect nodes
- Test interactive features

### Running E2E Tests

```bash
# Install Playwright browsers (one-time)
playwright install

# Run all E2E tests
pytest tests/test_flask_e2e_playwright.py

# Run specific test class
pytest tests/test_flask_e2e_playwright.py::TestInitializationWorkflow

# Run with headed mode (see browser)
pytest tests/test_flask_e2e_playwright.py --headed

# Run with specific browser
pytest tests/test_flask_e2e_playwright.py --browser chromium

# Run in debug mode
pytest tests/test_flask_e2e_playwright.py --debug
```

### Playwright Configuration

Create `pytest.ini` with:
```ini
[pytest]
asyncio_mode = auto
```

Or in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### Best Practices

1. **Wait for elements**: Use `expect(locator).to_be_visible()`
2. **Use meaningful selectors**: ID > Class > Text
3. **Test user flows**: Not implementation details
4. **Screenshot on failure**: Helps debug
5. **Headless by default**: Faster CI/CD execution

## Tier 3: Local Development with Hot Reload

### Purpose
- **Manual exploration**: Try new features interactively
- **Real API integration**: Test with actual Claude responses
- **Visual debugging**: See UI updates in real-time
- **Quick iteration**: Automatic reload on file changes

### Setup

1. **Install dependencies**:
```bash
pip install -r requirements-dev.txt
pip install -e .
```

2. **Set environment variables**:
```bash
export ANTHROPIC_API_KEY="your-key-here"
export FLASK_ENV=development
export FLASK_DEBUG=1
```

3. **Run Flask server**:
```bash
python3 app.py
```

The server starts on `http://localhost:5001` with hot reload enabled.

### Using the Development Server

1. **Enter a problem statement** in the web UI
2. **Click Initialize** to create the graph
3. **Click Supervisor Thinks** to get agent recommendations
4. **Watch the graph visualization** update in real-time
5. **Click Delegate** to assign tasks
6. **Get agent responses** by clicking response buttons
7. **Chat with supervisor** using the chat window

### Debugging Tips

**1. Check Flask logs**:
```
[INFO] __main__: Supervisor response: <response text>
[DEBUG] __main__: Message type: AssistantMessage
```

**2. Browser console (F12)**:
- Check JavaScript errors
- View API request/response payloads
- Monitor network activity

**3. Check graph state**:
Visit `http://localhost:5001/api/graph-state` to see current graph.

**4. Test individual endpoints**:
```bash
# Initialize
curl -X POST http://localhost:5001/api/initialize \
  -H "Content-Type: application/json" \
  -d '{"problem": "Your problem here"}'

# Get graph state
curl http://localhost:5001/api/graph-state
```

## Choosing the Right Testing Approach

### Use Unit Tests When:
- Testing specific API behavior
- Checking error handling
- Validating data transformations
- Need fast feedback (< 1 second)
- Testing in CI/CD pipeline

### Use E2E Tests When:
- Testing complete user workflows
- Checking UI updates correctly
- Verifying visual appearance
- Testing browser compatibility
- Need to validate real interactions

### Use Manual Testing When:
- Exploring new features
- Debugging complex issues
- Testing with real Claude API responses
- Getting visual feedback
- Manual acceptance testing

## CI/CD Integration

### GitHub Actions Workflow

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -e ".[dev]"
      - run: pytest tests/test_flask_api.py -v --cov=app

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -e ".[dev]"
      - run: playwright install
      - run: pytest tests/test_flask_e2e_playwright.py -v
```

### Pre-commit Hooks

See `.pre-commit-config.yaml` for automatic test execution before commits.

## Test Execution Commands

### Quick Testing (5-10 seconds)
```bash
# Run unit tests only
pytest tests/test_flask_api.py -q
```

### Full Testing (30-60 seconds)
```bash
# Run all tests
pytest tests/test_flask_api.py tests/test_flask_e2e_playwright.py -v
```

### With Coverage Report
```bash
pytest tests/test_flask_api.py --cov=app --cov-report=html
# Open htmlcov/index.html in browser
```

### E2E with Headed Mode
```bash
# See browser during test execution
pytest tests/test_flask_e2e_playwright.py --headed -v
```

## Metrics and Goals

### Coverage Targets
- **Unit tests**: 80%+ code coverage
- **E2E tests**: Cover all critical workflows
- **Total**: >85% combined coverage

### Performance Targets
- **Unit tests**: < 2 seconds each
- **E2E tests**: < 30 seconds per test
- **Full suite**: < 5 minutes

### Quality Goals
- Zero failing tests in main branch
- All PRs must pass tests before merge
- New features require new tests
- Regression bugs require new tests

## Troubleshooting

### E2E Tests Timeout
```bash
# Increase timeout
pytest tests/test_flask_e2e_playwright.py --timeout=60
```

### Flask Server Not Running
```bash
# Make sure app.py is running
python3 app.py

# Or use a fixture that starts the server
@pytest.fixture
def running_app(app):
    with app.test_client() as client:
        yield client
```

### Mock Not Working
```python
# Make sure to patch the right location
@patch("app.ClaudeSDKClient")  # Not "claude_agent_sdk.ClaudeSDKClient"
def test_something(mock_client):
    pass
```

### Playwright Browser Not Found
```bash
# Install browsers
playwright install chromium

# Or install all
playwright install
```

## Next Steps

1. **Increase coverage**: Aim for >90% in critical paths
2. **Add performance tests**: Monitor response times
3. **Add load tests**: Test with multiple concurrent users
4. **Visual regression testing**: Detect UI changes automatically
5. **Contract testing**: Validate API contracts with backend

## Related Documentation

- [Flask Testing Documentation](https://flask.palletsprojects.com/testing/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Playwright Documentation](https://playwright.dev/python/)
- [CI/CD Setup](./CI_CD_SETUP.md)
