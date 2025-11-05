"""
Unit tests for AgentSessionManager.

Tests cover:
- Session creation and caching
- Agent lifecycle (start, stop, restart)
- Error handling and recovery with retries
- Status tracking and management
"""

from unittest.mock import AsyncMock, patch

import pytest

from claude_agent_graph.exceptions import AgentGraphError, NodeNotFoundError
from claude_agent_graph.graph import AgentGraph
from claude_agent_graph.models import NodeStatus

# ==================== Fixtures ====================


@pytest.fixture
async def graph():
    """Create a test graph with some nodes."""
    g = AgentGraph(name="test_graph")
    await g.add_node("node1", "You are agent 1", model="claude-sonnet-4-20250514")
    await g.add_node("node2", "You are agent 2", model="claude-sonnet-4-20250514")
    await g.add_node("node3", "You are agent 3", model="claude-sonnet-4-20250514")
    return g


@pytest.fixture
async def agent_manager(graph):
    """Get the agent manager from the graph."""
    return graph._agent_manager


@pytest.fixture
def mock_claude_client():
    """Create a mock ClaudeSDKClient."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


# ==================== Session Creation Tests ====================


class TestSessionCreation:
    """Tests for agent session creation and caching."""

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_create_session_basic(
        self, mock_options_class, mock_client_class, agent_manager, mock_claude_client
    ):
        """Test basic session creation."""
        mock_client_class.return_value = mock_claude_client

        session = await agent_manager.create_session("node1")

        assert session is not None
        assert "node1" in agent_manager._sessions
        assert agent_manager._sessions["node1"] == mock_claude_client
        mock_options_class.assert_called_once()
        mock_client_class.assert_called_once()

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_create_session_with_metadata(
        self, mock_options_class, mock_client_class, graph, mock_claude_client
    ):
        """Test session creation with node metadata (working_directory -> cwd)."""
        await graph.add_node(
            "node_with_wd",
            "Test prompt",
            working_directory="/tmp/test",
        )
        mock_client_class.return_value = mock_claude_client

        await graph._agent_manager.create_session("node_with_wd")

        # Verify working_directory was mapped to cwd parameter in options
        call_kwargs = mock_options_class.call_args.kwargs
        assert call_kwargs.get("cwd") == "/tmp/test"

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_session_caching(
        self, mock_options_class, mock_client_class, agent_manager, mock_claude_client
    ):
        """Test that sessions are cached and reused."""
        mock_client_class.return_value = mock_claude_client

        session1 = await agent_manager.get_session("node1")
        session2 = await agent_manager.get_session("node1")

        assert session1 is session2
        # Should only be called once due to caching
        mock_client_class.assert_called_once()

    async def test_create_session_node_not_found(self, agent_manager):
        """Test session creation fails for nonexistent node."""
        with pytest.raises(NodeNotFoundError):
            await agent_manager.create_session("nonexistent")

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_create_session_uses_effective_prompt(
        self, mock_options_class, mock_client_class, graph, mock_claude_client
    ):
        """Test that effective_system_prompt is used if available."""
        node = graph.get_node("node1")
        node.effective_system_prompt = "Effective prompt with controller info"

        mock_client_class.return_value = mock_claude_client

        await graph._agent_manager.create_session("node1")

        # Verify effective prompt was used
        call_kwargs = mock_options_class.call_args.kwargs
        assert call_kwargs["system_prompt"] == "Effective prompt with controller info"


# ==================== Lifecycle Management Tests ====================


class TestLifecycleManagement:
    """Tests for agent lifecycle operations."""

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_start_agent(
        self, mock_options_class, mock_client_class, agent_manager, graph, mock_claude_client
    ):
        """Test starting an agent."""
        mock_client_class.return_value = mock_claude_client

        await agent_manager.start_agent("node1")

        node = graph.get_node("node1")
        assert node.status == NodeStatus.ACTIVE
        assert "node1" in agent_manager._contexts
        assert agent_manager.is_running("node1")
        mock_claude_client.__aenter__.assert_called_once()

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_stop_agent(
        self, mock_options_class, mock_client_class, agent_manager, graph, mock_claude_client
    ):
        """Test stopping an agent."""
        mock_client_class.return_value = mock_claude_client

        # Start then stop
        await agent_manager.start_agent("node1")
        await agent_manager.stop_agent("node1")

        node = graph.get_node("node1")
        assert node.status == NodeStatus.STOPPED
        assert "node1" not in agent_manager._contexts
        assert not agent_manager.is_running("node1")
        mock_claude_client.__aexit__.assert_called_once()

    async def test_stop_agent_not_running(self, agent_manager):
        """Test stopping an agent that isn't running (should be idempotent)."""
        # Should not raise an exception
        await agent_manager.stop_agent("node1")

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_restart_agent(
        self, mock_options_class, mock_client_class, agent_manager, graph, mock_claude_client
    ):
        """Test restarting an agent."""
        mock_client_class.return_value = mock_claude_client

        await agent_manager.start_agent("node1")
        await agent_manager.restart_agent("node1")

        node = graph.get_node("node1")
        assert node.status == NodeStatus.ACTIVE
        assert agent_manager.is_running("node1")
        # Should call __aexit__ (stop) and __aenter__ (start)
        assert mock_claude_client.__aexit__.call_count == 1
        assert mock_claude_client.__aenter__.call_count == 2  # Once for start, once for restart

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_start_already_running_raises(
        self, mock_options_class, mock_client_class, agent_manager, mock_claude_client
    ):
        """Test starting an already running agent raises error."""
        mock_client_class.return_value = mock_claude_client

        await agent_manager.start_agent("node1")

        with pytest.raises(AgentGraphError, match="already running"):
            await agent_manager.start_agent("node1")

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_status_transitions(
        self, mock_options_class, mock_client_class, agent_manager, graph, mock_claude_client
    ):
        """Test node status transitions through lifecycle."""
        mock_client_class.return_value = mock_claude_client

        node = graph.get_node("node1")

        # Initial state
        assert node.status == NodeStatus.INITIALIZING

        # After start
        await agent_manager.start_agent("node1")
        assert node.status == NodeStatus.ACTIVE

        # After stop
        await agent_manager.stop_agent("node1")
        assert node.status == NodeStatus.STOPPED

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_multiple_concurrent_agents(
        self, mock_options_class, mock_client_class, agent_manager, graph
    ):
        """Test managing multiple agents concurrently."""
        # Create separate mocks for each agent
        mock1 = AsyncMock()
        mock1.__aenter__ = AsyncMock(return_value=mock1)
        mock1.__aexit__ = AsyncMock(return_value=None)

        mock2 = AsyncMock()
        mock2.__aenter__ = AsyncMock(return_value=mock2)
        mock2.__aexit__ = AsyncMock(return_value=None)

        mock3 = AsyncMock()
        mock3.__aenter__ = AsyncMock(return_value=mock3)
        mock3.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.side_effect = [mock1, mock2, mock3]

        await agent_manager.start_agent("node1")
        await agent_manager.start_agent("node2")
        await agent_manager.start_agent("node3")

        assert agent_manager.get_running_count() == 3
        assert set(agent_manager.get_running_agents()) == {"node1", "node2", "node3"}

        # All nodes should be ACTIVE
        assert graph.get_node("node1").status == NodeStatus.ACTIVE
        assert graph.get_node("node2").status == NodeStatus.ACTIVE
        assert graph.get_node("node3").status == NodeStatus.ACTIVE

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_stop_all_cleanup(
        self, mock_options_class, mock_client_class, agent_manager, graph
    ):
        """Test stop_all cleans up all running agents."""
        # Create separate mocks
        mocks = []
        for _ in range(3):
            m = AsyncMock()
            m.__aenter__ = AsyncMock(return_value=m)
            m.__aexit__ = AsyncMock(return_value=None)
            mocks.append(m)

        mock_client_class.side_effect = mocks

        # Start multiple agents
        await agent_manager.start_agent("node1")
        await agent_manager.start_agent("node2")
        await agent_manager.start_agent("node3")

        # Stop all
        await agent_manager.stop_all()

        assert agent_manager.get_running_count() == 0
        assert all(
            graph.get_node(nid).status == NodeStatus.STOPPED for nid in ["node1", "node2", "node3"]
        )


# ==================== Error Recovery Tests ====================


class TestErrorRecovery:
    """Tests for error handling and retry logic."""

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_retry_on_transient_error(
        self, mock_options_class, mock_client_class, agent_manager, graph
    ):
        """Test retry logic succeeds after transient error."""
        mock_client = AsyncMock()

        # Fail once, then succeed
        mock_client.__aenter__ = AsyncMock(side_effect=[Exception("Network error"), mock_client])
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        await agent_manager.start_agent_with_recovery("node1")

        node = graph.get_node("node1")
        assert node.status == NodeStatus.ACTIVE
        assert "last_error" not in node.metadata
        assert agent_manager.is_running("node1")

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    @patch("claude_agent_graph.agent_manager.asyncio.sleep", new_callable=AsyncMock)
    async def test_exponential_backoff(
        self, mock_sleep, mock_options_class, mock_client_class, agent_manager
    ):
        """Test exponential backoff delays."""
        mock_client = AsyncMock()
        # Fail twice, succeed on third attempt
        mock_client.__aenter__ = AsyncMock(
            side_effect=[
                Exception("Error 1"),
                Exception("Error 2"),
                mock_client,
            ]
        )
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        await agent_manager.start_agent_with_recovery("node1")

        # Verify backoff delays: 1s, 2s
        assert mock_sleep.call_count == 2
        calls = mock_sleep.call_args_list
        assert calls[0][0][0] == 1.0  # First retry: 1 second
        assert calls[1][0][0] == 2.0  # Second retry: 2 seconds

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_max_retries_exceeded(
        self, mock_options_class, mock_client_class, agent_manager, graph
    ):
        """Test that max retries are respected."""
        mock_client = AsyncMock()
        # Always fail
        mock_client.__aenter__ = AsyncMock(side_effect=Exception("Permanent error"))
        mock_client_class.return_value = mock_client

        with pytest.raises(AgentGraphError, match="failed after 3 retries"):
            await agent_manager.start_agent_with_recovery("node1")

        node = graph.get_node("node1")
        assert node.status == NodeStatus.ERROR

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_permanent_error_status(
        self, mock_options_class, mock_client_class, agent_manager, graph
    ):
        """Test that permanent errors set ERROR status."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(side_effect=Exception("Fatal error"))
        mock_client_class.return_value = mock_client

        with pytest.raises(AgentGraphError):
            await agent_manager.start_agent_with_recovery("node1")

        node = graph.get_node("node1")
        assert node.status == NodeStatus.ERROR

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_error_metadata_stored(
        self, mock_options_class, mock_client_class, agent_manager, graph
    ):
        """Test that error details are stored in metadata."""
        mock_client = AsyncMock()
        error_msg = "Authentication failed"
        mock_client.__aenter__ = AsyncMock(side_effect=Exception(error_msg))
        mock_client_class.return_value = mock_client

        with pytest.raises(AgentGraphError):
            await agent_manager.start_agent_with_recovery("node1")

        node = graph.get_node("node1")
        assert "last_error" in node.metadata
        assert error_msg in node.metadata["last_error"]
        assert node.metadata.get("error_count", 0) >= 1

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_recovery_after_error(
        self, mock_options_class, mock_client_class, agent_manager, graph
    ):
        """Test agent can recover after an error."""
        mock_client = AsyncMock()

        # Fail on first attempt
        mock_client.__aenter__ = AsyncMock(side_effect=Exception("Error"))
        mock_client_class.return_value = mock_client

        with pytest.raises(AgentGraphError):
            await agent_manager.start_agent_with_recovery("node1")

        assert graph.get_node("node1").status == NodeStatus.ERROR

        # Succeed on next attempt
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)

        await agent_manager.start_agent_with_recovery("node1")

        node = graph.get_node("node1")
        assert node.status == NodeStatus.ACTIVE
        # Error metadata should be cleared
        assert "last_error" not in node.metadata

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_error_state_isolation(
        self, mock_options_class, mock_client_class, agent_manager, graph
    ):
        """Test that errors in one agent don't affect others."""
        mock_client1 = AsyncMock()
        mock_client1.__aenter__ = AsyncMock(side_effect=Exception("Error"))

        mock_client2 = AsyncMock()
        mock_client2.__aenter__ = AsyncMock(return_value=mock_client2)
        mock_client2.__aexit__ = AsyncMock(return_value=None)

        mock_client_class.side_effect = [mock_client1, mock_client2]

        # node1 fails
        with pytest.raises(AgentGraphError):
            await agent_manager.start_agent_with_recovery("node1")

        # node2 should succeed
        await agent_manager.start_agent_with_recovery("node2")

        assert graph.get_node("node1").status == NodeStatus.ERROR
        assert graph.get_node("node2").status == NodeStatus.ACTIVE

    async def test_get_running_agents(self, agent_manager):
        """Test getting list of running agents."""
        assert agent_manager.get_running_agents() == []

    async def test_get_session_count(self, agent_manager):
        """Test getting session count."""
        assert agent_manager.get_session_count() == 0

    async def test_get_running_count(self, agent_manager):
        """Test getting running count."""
        assert agent_manager.get_running_count() == 0


# ==================== Helper Method Tests ====================


class TestHelperMethods:
    """Tests for helper methods."""

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_is_running(
        self, mock_options_class, mock_client_class, agent_manager, mock_claude_client
    ):
        """Test is_running check."""
        mock_client_class.return_value = mock_claude_client

        assert not agent_manager.is_running("node1")

        await agent_manager.start_agent("node1")
        assert agent_manager.is_running("node1")

        await agent_manager.stop_agent("node1")
        assert not agent_manager.is_running("node1")

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_get_running_agents_list(
        self, mock_options_class, mock_client_class, agent_manager
    ):
        """Test getting list of running agents."""
        # Create separate mocks
        mocks = []
        for _ in range(2):
            m = AsyncMock()
            m.__aenter__ = AsyncMock(return_value=m)
            m.__aexit__ = AsyncMock(return_value=None)
            mocks.append(m)

        mock_client_class.side_effect = mocks

        await agent_manager.start_agent("node1")
        await agent_manager.start_agent("node2")

        running = agent_manager.get_running_agents()
        assert set(running) == {"node1", "node2"}

    @patch("claude_agent_graph.agent_manager.ClaudeSDKClient")
    @patch("claude_agent_graph.agent_manager.ClaudeAgentOptions")
    async def test_get_session_and_running_counts(
        self, mock_options_class, mock_client_class, agent_manager
    ):
        """Test session and running count methods."""
        # Create mocks
        mocks = []
        for _ in range(3):
            m = AsyncMock()
            m.__aenter__ = AsyncMock(return_value=m)
            m.__aexit__ = AsyncMock(return_value=None)
            mocks.append(m)

        mock_client_class.side_effect = mocks

        # Start 3 agents
        await agent_manager.start_agent("node1")
        await agent_manager.start_agent("node2")
        await agent_manager.start_agent("node3")

        assert agent_manager.get_session_count() == 3
        assert agent_manager.get_running_count() == 3

        # Stop one
        await agent_manager.stop_agent("node2")

        assert agent_manager.get_session_count() == 3  # Sessions still cached
        assert agent_manager.get_running_count() == 2  # Only 2 running
