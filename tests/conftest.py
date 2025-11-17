"""
Pytest configuration and fixtures for Flask application testing.

This module provides:
- Flask app fixture for testing
- Mock Claude API responses
- Database/storage fixtures
- AgentCollaborationManager fixtures
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flask import Flask

# Ensure project root is in Python path for importing app module
_project_root = Path(__file__).parent.parent
_app_path = _project_root / "app.py"

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Debug: verify app.py exists
if not _app_path.exists():
    raise FileNotFoundError(
        f"app.py not found at {_app_path.absolute()}. "
        f"Project root: {_project_root.absolute()}"
    )

try:
    from app import AgentCollaborationManager, app as flask_app
except ImportError as e:
    # Fallback: try absolute import using importlib
    import importlib.util
    try:
        app_spec = importlib.util.spec_from_file_location("app", str(_app_path))
        if app_spec and app_spec.loader:
            app_module = importlib.util.module_from_spec(app_spec)
            sys.modules["app"] = app_module  # Register module before exec
            app_spec.loader.exec_module(app_module)
            flask_app = app_module.app
            AgentCollaborationManager = app_module.AgentCollaborationManager
        else:
            raise ImportError(
                f"Could not create spec for app module at {_app_path}"
            ) from e
    except Exception as import_error:
        raise ImportError(
            f"Failed to import app module from {_app_path.absolute()}: {import_error}"
        ) from import_error


# ============================================================================
# Flask App Fixtures
# ============================================================================


@pytest.fixture
def app() -> Flask:
    """
    Create and configure Flask app for testing.

    Returns:
        Flask: Configured Flask test app
    """
    flask_app.config["TESTING"] = True
    flask_app.config["JSON_SORT_KEYS"] = False

    yield flask_app


@pytest.fixture
def client(app: Flask):
    """
    Create Flask test client.

    Args:
        app: Flask app fixture

    Returns:
        FlaskClient: Test client for making requests
    """
    return app.test_client()


# ============================================================================
# Manager Fixtures
# ============================================================================


@pytest.fixture
def manager():
    """
    Get the global AgentCollaborationManager used by Flask app.

    Returns:
        AgentCollaborationManager: Global manager instance
    """
    # Get the manager from the app module that was imported at the top
    app_module = sys.modules.get('app')
    if app_module and hasattr(app_module, 'manager'):
        return app_module.manager
    # Fallback: raise error if not found
    raise RuntimeError("AgentCollaborationManager not found in app module")


# ============================================================================
# Mock API Response Fixtures
# ============================================================================


@pytest.fixture
def mock_supervisor_analysis() -> dict[str, Any]:
    """
    Provide mock supervisor analysis response.

    Returns:
        dict: Sample supervisor analysis with agents needed
    """
    return {
        "analysis": "This is a complex problem requiring multiple specialist perspectives.",
        "agents_needed": [
            {
                "name": "researcher",
                "role": "researcher",
                "description": "Conduct in-depth research on the topic",
            },
            {
                "name": "analyst",
                "role": "analyst",
                "description": "Analyze research findings and identify patterns",
            },
            {
                "name": "strategist",
                "role": "strategist",
                "description": "Develop strategic recommendations",
            },
        ],
        "approach": "Systematic problem-solving with specialized agent roles",
    }


@pytest.fixture
def mock_delegation_response() -> dict[str, Any]:
    """
    Provide mock task delegation response.

    Returns:
        dict: Sample delegations from supervisor to agents
    """
    return {
        "delegations": [
            {
                "agent": "researcher",
                "task": "Research current best practices and solutions",
            },
            {
                "agent": "analyst",
                "task": "Analyze the research and identify key insights",
            },
            {
                "agent": "strategist",
                "task": "Develop a strategic implementation plan",
            },
        ]
    }


@pytest.fixture
def mock_agent_response() -> str:
    """
    Provide mock agent response text.

    Returns:
        str: Sample agent response
    """
    return "Here is my analysis and contribution to solving this problem..."


@pytest.fixture
def mock_supervisor_chat_response() -> str:
    """
    Provide mock supervisor chat response.

    Returns:
        str: Sample supervisor chat response
    """
    return "That's a great question. Let me help clarify our approach..."


# ============================================================================
# AsyncIO and Event Loop Fixtures
# ============================================================================


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Create and provide event loop for async tests.

    Yields:
        asyncio.AbstractEventLoop: Event loop for async operations
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ============================================================================
# Mock ClaudeSDKClient Fixtures
# ============================================================================


@pytest.fixture
def mock_claude_client(
    mock_supervisor_analysis: dict[str, Any],
) -> AsyncMock:
    """
    Create mock ClaudeSDKClient for supervisor.

    Args:
        mock_supervisor_analysis: Mock analysis response

    Returns:
        AsyncMock: Mocked ClaudeSDKClient
    """

    async def async_generator_mock(analysis_response: dict[str, Any]):
        """Create async generator that yields message."""

        class MessageMock:
            def __init__(self, content: str):
                self.content = content

        # Create mock message
        message = MessageMock(json.dumps(analysis_response))
        message.__class__.__name__ = "AssistantMessage"
        yield message

    client = AsyncMock()
    client.query = AsyncMock()
    # Return the async generator directly instead of wrapping in AsyncMock
    client.receive_messages = lambda: async_generator_mock(mock_supervisor_analysis)

    # Make the context manager work
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    return client


@pytest.fixture
def mock_agent_client(mock_agent_response: str) -> AsyncMock:
    """
    Create mock ClaudeSDKClient for agent.

    Args:
        mock_agent_response: Mock agent response text

    Returns:
        AsyncMock: Mocked ClaudeSDKClient for agent
    """

    async def async_generator_mock(response_text: str):
        """Create async generator that yields message."""

        class MessageMock:
            def __init__(self, content: str):
                self.content = content

        # Create mock message
        message = MessageMock(response_text)
        message.__class__.__name__ = "AssistantMessage"
        yield message

    client = AsyncMock()
    client.query = AsyncMock()
    # Return the async generator directly instead of wrapping in AsyncMock
    client.receive_messages = lambda: async_generator_mock(mock_agent_response)

    # Make the context manager work
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    return client


# ============================================================================
# Request/Response JSON Fixtures
# ============================================================================


@pytest.fixture
def initialize_request_payload() -> dict[str, str]:
    """
    Provide sample initialization request payload.

    Returns:
        dict: Request payload for /api/initialize endpoint
    """
    return {
        "problem": "How should we approach building a scalable web application?"
    }


@pytest.fixture
def supervisor_think_request_payload() -> dict[str, str]:
    """
    Provide sample supervisor think request payload.

    Returns:
        dict: Request payload for /api/supervisor-think endpoint
    """
    return {
        "problem": "What is the best approach to system architecture design?"
    }


@pytest.fixture
def delegate_request_payload() -> dict[str, str]:
    """
    Provide sample delegation request payload.

    Returns:
        dict: Request payload for /api/delegate endpoint
    """
    return {"problem": "Analyze and propose solutions for scaling challenges"}


@pytest.fixture
def agent_response_request_payload() -> dict[str, str]:
    """
    Provide sample agent response request payload.

    Returns:
        dict: Request payload for /api/agent-response endpoint
    """
    return {
        "agent": "researcher",
        "task": "Research current best practices for system design",
    }


@pytest.fixture
def supervisor_chat_request_payload() -> dict[str, str]:
    """
    Provide sample supervisor chat request payload.

    Returns:
        dict: Request payload for /api/supervisor-chat endpoint
    """
    return {"message": "Can you clarify your approach to this problem?"}


# ============================================================================
# Mocked ClaudeSDKClient Patch Fixtures
# ============================================================================


@pytest.fixture
def patch_claude_sdk(mock_claude_client: AsyncMock):
    """
    Patch ClaudeSDKClient for testing.

    Args:
        mock_claude_client: Mock client fixture

    Yields:
        MagicMock: Patched ClaudeSDKClient class
    """
    with patch("app.ClaudeSDKClient", return_value=mock_claude_client) as mock:
        yield mock


@pytest.fixture
def patch_agent_graph(manager: AgentCollaborationManager):
    """
    Patch AgentGraph class for testing and initialize manager's graph.

    Args:
        manager: AgentCollaborationManager instance

    Yields:
        MagicMock: Patched AgentGraph class
    """
    with patch("app.AgentGraph") as mock:
        # Setup async methods
        mock_instance = AsyncMock()
        mock_instance.add_node = AsyncMock()
        mock_instance.add_edge = AsyncMock()
        mock_instance.get_nodes = MagicMock(
            return_value=[
                MagicMock(
                    node_id="supervisor",
                    status=MagicMock(value="active"),
                    model="claude-haiku-4-5-20251001",
                    metadata={},
                ),
                MagicMock(
                    node_id="researcher",
                    status=MagicMock(value="active"),
                    model="claude-haiku-4-5-20251001",
                    metadata={"role": "researcher"},
                ),
            ]
        )
        mock_instance._edges = {}

        mock.return_value = mock_instance
        # Initialize the manager's graph with the mocked instance
        manager.graph = mock_instance
        yield mock


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_problem_statement() -> str:
    """
    Provide a sample problem statement for testing.

    Returns:
        str: Sample problem statement
    """
    return (
        "How can we design a microservices architecture "
        "that scales to handle 1 million concurrent users?"
    )


@pytest.fixture
def incomplete_problem_statement() -> str:
    """
    Provide an incomplete problem statement (edge case).

    Returns:
        str: Incomplete problem statement
    """
    return "Get the vote of 3 different agents on what the pri..."


@pytest.fixture
def complex_problem_statement() -> str:
    """
    Provide a complex, detailed problem statement.

    Returns:
        str: Complex problem statement
    """
    return (
        "We need to build a real-time collaborative platform that:"
        "\n1. Supports 10,000+ concurrent users"
        "\n2. Requires sub-100ms latency for interactions"
        "\n3. Must handle complex data transformations"
        "\n4. Needs advanced security and compliance features"
        "\n5. Should be cost-effective and maintainable"
    )


@pytest.fixture
def graph_state_response() -> dict[str, Any]:
    """
    Provide sample graph state response.

    Returns:
        dict: Sample graph state with nodes and links
    """
    return {
        "nodes": [
            {
                "id": "supervisor",
                "status": "active",
                "model": "claude-haiku-4-5-20251001",
                "metadata": {},
            },
            {
                "id": "researcher",
                "status": "active",
                "model": "claude-haiku-4-5-20251001",
                "metadata": {"role": "researcher"},
            },
            {
                "id": "analyst",
                "status": "active",
                "model": "claude-haiku-4-5-20251001",
                "metadata": {"role": "analyst"},
            },
        ],
        "links": [
            {"source": "supervisor", "target": "researcher", "directed": False},
            {"source": "supervisor", "target": "analyst", "directed": False},
            {"source": "researcher", "target": "analyst", "directed": False},
        ],
    }


# ============================================================================
# Cleanup and Teardown Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_flask_context():
    """
    Automatically cleanup Flask app context after each test.

    Yields:
        None
    """
    yield
    # Cleanup is handled automatically by Flask test client


@pytest.fixture
def temporary_storage_path(tmp_path: Path) -> Path:
    """
    Provide a temporary storage path for tests.

    Args:
        tmp_path: Pytest temporary directory

    Returns:
        Path: Temporary storage path
    """
    storage_dir = tmp_path / "graph_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir
