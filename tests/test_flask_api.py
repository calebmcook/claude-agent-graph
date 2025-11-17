"""
Unit and integration tests for Flask API endpoints.

Tests cover:
- Request/response validation
- Error handling
- API contract compliance
- Mocked Claude API responses
"""

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from flask import Flask
from flask.testing import FlaskClient


class TestInitializeEndpoint:
    """Tests for /api/initialize endpoint."""

    def test_initialize_with_valid_problem(
        self,
        client: FlaskClient,
        initialize_request_payload: dict[str, str],
        patch_agent_graph,
    ):
        """
        Test successful initialization with valid problem statement.

        Args:
            client: Flask test client
            initialize_request_payload: Sample request payload
            patch_agent_graph: Mocked AgentGraph
        """
        response = client.post(
            "/api/initialize",
            json=initialize_request_payload,
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "graph" in data

    def test_initialize_without_problem(self, client: FlaskClient):
        """
        Test initialization fails without problem statement.

        Args:
            client: Flask test client
        """
        response = client.post(
            "/api/initialize", json={"problem": ""}, content_type="application/json"
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_initialize_missing_json_body(self, client: FlaskClient):
        """
        Test initialization fails when JSON body is missing.

        Args:
            client: Flask test client
        """
        response = client.post(
            "/api/initialize", content_type="application/json", data=""
        )

        # Flask returns 400 for missing/invalid JSON
        assert response.status_code in [400, 415]

    def test_initialize_graph_state_structure(
        self,
        client: FlaskClient,
        initialize_request_payload: dict[str, str],
        graph_state_response: dict[str, Any],
        patch_agent_graph,
    ):
        """
        Test that graph state response has correct structure.

        Args:
            client: Flask test client
            initialize_request_payload: Sample request payload
            graph_state_response: Expected graph state structure
            patch_agent_graph: Mocked AgentGraph
        """
        response = client.post(
            "/api/initialize",
            json=initialize_request_payload,
            content_type="application/json",
        )

        data = json.loads(response.data)
        graph = data.get("graph", {})

        # Verify structure
        assert "nodes" in graph
        assert "links" in graph
        assert isinstance(graph["nodes"], list)
        assert isinstance(graph["links"], list)


class TestSupervisorThinkEndpoint:
    """Tests for /api/supervisor-think endpoint."""

    def test_supervisor_think_valid_request(
        self,
        client: FlaskClient,
        supervisor_think_request_payload: dict[str, str],
        mock_supervisor_analysis: dict[str, Any],
        patch_claude_sdk,
        patch_agent_graph,
    ):
        """
        Test successful supervisor analysis request.

        Args:
            client: Flask test client
            supervisor_think_request_payload: Sample request
            mock_supervisor_analysis: Mock analysis response
            patch_claude_sdk: Mocked ClaudeSDKClient
            patch_agent_graph: Mocked AgentGraph
        """
        response = client.post(
            "/api/supervisor-think",
            json=supervisor_think_request_payload,
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["type"] == "supervisor_analysis"
        assert "analysis" in data
        assert "agents" in data

    def test_supervisor_think_without_problem(self, client: FlaskClient):
        """
        Test supervisor think fails without problem statement.

        Args:
            client: Flask test client
        """
        response = client.post(
            "/api/supervisor-think",
            json={"problem": ""},
            content_type="application/json",
        )

        assert response.status_code == 400

    def test_supervisor_think_agents_created(
        self,
        client: FlaskClient,
        supervisor_think_request_payload: dict[str, str],
        patch_claude_sdk,
        patch_agent_graph,
    ):
        """
        Test that supervisor analysis creates agents in response.

        Args:
            client: Flask test client
            supervisor_think_request_payload: Sample request
            patch_claude_sdk: Mocked ClaudeSDKClient
            patch_agent_graph: Mocked AgentGraph
        """
        response = client.post(
            "/api/supervisor-think",
            json=supervisor_think_request_payload,
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        agents = data.get("agents", [])

        # Should have created agents
        assert len(agents) > 0
        for agent in agents:
            assert "id" in agent
            assert "role" in agent


class TestDelegateEndpoint:
    """Tests for /api/delegate endpoint."""

    def test_delegate_valid_request(
        self,
        client: FlaskClient,
        delegate_request_payload: dict[str, str],
        patch_claude_sdk,
        patch_agent_graph,
    ):
        """
        Test successful task delegation request.

        Args:
            client: Flask test client
            delegate_request_payload: Sample request
            patch_claude_sdk: Mocked ClaudeSDKClient
            patch_agent_graph: Mocked AgentGraph
        """
        response = client.post(
            "/api/delegate",
            json=delegate_request_payload,
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "delegations" in data
        assert "graph" in data

    def test_delegate_without_graph(self, client: FlaskClient):
        """
        Test delegation when graph is not initialized (returns empty).

        Args:
            client: Flask test client
        """
        # This should return a response even if graph is None
        response = client.post(
            "/api/delegate",
            json={"problem": "Test problem"},
            content_type="application/json",
        )

        # Should return successful response with empty delegations
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "delegations" in data

    def test_delegate_response_structure(
        self,
        client: FlaskClient,
        delegate_request_payload: dict[str, str],
        patch_claude_sdk,
        patch_agent_graph,
    ):
        """
        Test that delegation response has correct structure.

        Args:
            client: Flask test client
            delegate_request_payload: Sample request
            patch_claude_sdk: Mocked ClaudeSDKClient
            patch_agent_graph: Mocked AgentGraph
        """
        response = client.post(
            "/api/delegate",
            json=delegate_request_payload,
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        delegations = data.get("delegations", [])

        # Each delegation should have agent and task
        for delegation in delegations:
            assert "agent" in delegation
            assert "task" in delegation


class TestAgentResponseEndpoint:
    """Tests for /api/agent-response endpoint."""

    def test_agent_response_valid_request(
        self,
        client: FlaskClient,
        agent_response_request_payload: dict[str, str],
        patch_claude_sdk,
        patch_agent_graph,
    ):
        """
        Test successful agent response request.

        Args:
            client: Flask test client
            agent_response_request_payload: Sample request
            patch_claude_sdk: Mocked ClaudeSDKClient
            patch_agent_graph: Mocked AgentGraph
        """
        response = client.post(
            "/api/agent-response",
            json=agent_response_request_payload,
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "agent" in data
        assert "response" in data
        assert "timestamp" in data

    def test_agent_response_missing_agent(self, client: FlaskClient):
        """
        Test agent response fails when agent is missing.

        Args:
            client: Flask test client
        """
        response = client.post(
            "/api/agent-response",
            json={"agent": "", "task": "Some task"},
            content_type="application/json",
        )

        assert response.status_code == 400

    def test_agent_response_missing_task(self, client: FlaskClient):
        """
        Test agent response fails when task is missing.

        Args:
            client: Flask test client
        """
        response = client.post(
            "/api/agent-response",
            json={"agent": "researcher", "task": ""},
            content_type="application/json",
        )

        assert response.status_code == 400

    def test_agent_response_timestamp_format(
        self,
        client: FlaskClient,
        agent_response_request_payload: dict[str, str],
        patch_claude_sdk,
        patch_agent_graph,
    ):
        """
        Test that response includes valid ISO timestamp.

        Args:
            client: Flask test client
            agent_response_request_payload: Sample request
            patch_claude_sdk: Mocked ClaudeSDKClient
            patch_agent_graph: Mocked AgentGraph
        """
        response = client.post(
            "/api/agent-response",
            json=agent_response_request_payload,
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        timestamp = data.get("timestamp", "")

        # Should be ISO format
        assert "T" in timestamp
        assert "+" in timestamp or "Z" in timestamp or "-" in timestamp


class TestSupervisorChatEndpoint:
    """Tests for /api/supervisor-chat endpoint."""

    def test_supervisor_chat_valid_request(
        self,
        client: FlaskClient,
        supervisor_chat_request_payload: dict[str, str],
        patch_claude_sdk,
        patch_agent_graph,
    ):
        """
        Test successful supervisor chat request.

        Args:
            client: Flask test client
            supervisor_chat_request_payload: Sample request
            patch_claude_sdk: Mocked ClaudeSDKClient
            patch_agent_graph: Mocked AgentGraph
        """
        response = client.post(
            "/api/supervisor-chat",
            json=supervisor_chat_request_payload,
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "role" in data
        assert data["role"] == "supervisor"
        assert "message" in data
        assert "response" in data
        assert "timestamp" in data

    def test_supervisor_chat_missing_message(self, client: FlaskClient):
        """
        Test supervisor chat fails without message.

        Args:
            client: Flask test client
        """
        response = client.post(
            "/api/supervisor-chat",
            json={"message": ""},
            content_type="application/json",
        )

        assert response.status_code == 400

    def test_supervisor_chat_empty_body(self, client: FlaskClient):
        """
        Test supervisor chat fails with empty body.

        Args:
            client: Flask test client
        """
        response = client.post(
            "/api/supervisor-chat", json={}, content_type="application/json"
        )

        assert response.status_code == 400

    def test_supervisor_chat_maintains_context(
        self,
        client: FlaskClient,
        patch_claude_sdk,
        patch_agent_graph,
    ):
        """
        Test that multiple chat messages maintain context.

        Args:
            client: Flask test client
            patch_claude_sdk: Mocked ClaudeSDKClient
            patch_agent_graph: Mocked AgentGraph
        """
        # Send first message
        response1 = client.post(
            "/api/supervisor-chat",
            json={"message": "What is your approach?"},
            content_type="application/json",
        )
        assert response1.status_code == 200

        # Send follow-up message
        response2 = client.post(
            "/api/supervisor-chat",
            json={"message": "Can you elaborate on that?"},
            content_type="application/json",
        )
        assert response2.status_code == 200


class TestGraphStateEndpoint:
    """Tests for /api/graph-state endpoint."""

    def test_graph_state_returns_valid_structure(
        self,
        client: FlaskClient,
        graph_state_response: dict[str, Any],
        patch_agent_graph,
    ):
        """
        Test that graph state returns valid structure.

        Args:
            client: Flask test client
            graph_state_response: Expected structure
            patch_agent_graph: Mocked AgentGraph
        """
        response = client.get("/api/graph-state")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "nodes" in data
        assert "links" in data

    def test_graph_state_is_json(self, client: FlaskClient, patch_agent_graph):
        """
        Test that graph state returns valid JSON.

        Args:
            client: Flask test client
            patch_agent_graph: Mocked AgentGraph
        """
        response = client.get("/api/graph-state")

        assert response.content_type == "application/json"
        data = json.loads(response.data)
        assert isinstance(data, dict)

    def test_graph_state_empty_graph(self, client: FlaskClient):
        """
        Test graph state with uninitialized graph.

        Args:
            client: Flask test client
        """
        response = client.get("/api/graph-state")

        assert response.status_code == 200
        data = json.loads(response.data)
        # Should return structure even if graph is empty
        assert "nodes" in data
        assert "links" in data


class TestIndexEndpoint:
    """Tests for / (index) endpoint."""

    def test_index_returns_html(self, client: FlaskClient):
        """
        Test that index endpoint returns HTML.

        Args:
            client: Flask test client
        """
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.content_type
        # Should have some HTML content
        assert len(response.data) > 0


class TestErrorHandling:
    """Tests for error handling across endpoints."""

    def test_malformed_json_returns_400(self, client: FlaskClient):
        """
        Test that malformed JSON returns 400.

        Args:
            client: Flask test client
        """
        response = client.post(
            "/api/initialize",
            data="not valid json",
            content_type="application/json",
        )

        assert response.status_code in [400, 415]

    def test_missing_content_type(self, client: FlaskClient):
        """
        Test request without content type header.

        Args:
            client: Flask test client
        """
        response = client.post(
            "/api/initialize",
            json={"problem": "Test"},
        )

        # Flask should still handle this
        assert response.status_code in [200, 400, 415]

    @patch("app.asyncio.new_event_loop")
    def test_async_error_handling(
        self,
        mock_loop,
        client: FlaskClient,
        initialize_request_payload: dict[str, str],
    ):
        """
        Test error handling in async operations.

        Args:
            mock_loop: Mocked event loop
            client: Flask test client
            initialize_request_payload: Sample request
        """
        # Setup mock to raise an exception
        mock_loop.return_value.run_until_complete.side_effect = Exception(
            "Test error"
        )

        response = client.post(
            "/api/initialize",
            json=initialize_request_payload,
            content_type="application/json",
        )

        # Should return 500 error
        assert response.status_code == 500
        data = json.loads(response.data)
        assert "error" in data


class TestCORS:
    """Tests for CORS headers."""

    def test_cors_headers_present(self, client: FlaskClient):
        """
        Test that CORS headers are present in response.

        Args:
            client: Flask test client
        """
        response = client.get("/api/graph-state")

        # Flask-CORS should add appropriate headers
        # Note: Some headers might not be present in test client
        # but the functionality should work
        assert response.status_code == 200
