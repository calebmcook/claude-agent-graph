"""
Agent session management for claude-agent-graph.

This module manages ClaudeSDKClient instances for agent nodes, handling
lifecycle (start, stop, restart) and error recovery with retry logic.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from .exceptions import AgentGraphError, NodeNotFoundError
from .models import NodeStatus

if TYPE_CHECKING:
    from .graph import AgentGraph

# Import claude-agent-sdk with graceful handling
try:
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
except ImportError:
    # Allow module to load even if SDK not installed (for testing)
    ClaudeAgentOptions = Any  # type: ignore
    ClaudeSDKClient = Any  # type: ignore

logger = logging.getLogger(__name__)


class AgentSessionManager:
    """
    Manages ClaudeSDKClient instances for agent nodes.

    Handles agent lifecycle (creation, start, stop, restart) and provides
    error recovery with exponential backoff retry logic.
    """

    # Error recovery configuration
    MAX_RETRIES = 3
    RETRY_DELAYS = [1.0, 2.0, 4.0]  # Exponential backoff in seconds

    def __init__(self, graph: "AgentGraph") -> None:
        """
        Initialize the agent session manager.

        Args:
            graph: Reference to the parent AgentGraph instance
        """
        self._graph = graph
        self._sessions: dict[str, Any] = {}  # node_id -> ClaudeSDKClient
        self._contexts: dict[str, Any] = {}  # node_id -> active async context

    async def create_session(self, node_id: str) -> Any:
        """
        Create a ClaudeSDKClient for a node.

        Args:
            node_id: ID of the node

        Returns:
            Initialized ClaudeSDKClient instance

        Raises:
            NodeNotFoundError: If node doesn't exist
            AgentGraphError: If session creation fails
        """
        node = self._graph.get_node(node_id)

        # Use effective prompt if available (with controller injection),
        # otherwise use original system prompt
        prompt = node.effective_system_prompt or node.system_prompt

        try:
            options = ClaudeAgentOptions(
                system_prompt=prompt,
                model=node.model,
            )

            client = ClaudeSDKClient(options=options)
            self._sessions[node_id] = client

            logger.info(f"Created session for agent '{node_id}' with model '{node.model}'")
            return client

        except Exception as e:
            logger.error(f"Failed to create session for agent '{node_id}': {e}")
            raise AgentGraphError(f"Failed to create session for agent '{node_id}': {e}") from e

    async def get_session(self, node_id: str) -> Any:
        """
        Get existing session or create new session for a node.

        Args:
            node_id: ID of the node

        Returns:
            ClaudeSDKClient instance

        Raises:
            NodeNotFoundError: If node doesn't exist
            AgentGraphError: If session creation fails
        """
        if node_id not in self._sessions:
            await self.create_session(node_id)
        return self._sessions.get(node_id)

    async def start_agent(self, node_id: str) -> None:
        """
        Start an agent session.

        Enters the async context manager for the ClaudeSDKClient.
        Updates Node.status to ACTIVE.

        Args:
            node_id: ID of the node to start

        Raises:
            AgentGraphError: If agent is already running or start fails
            NodeNotFoundError: If node doesn't exist
        """
        if node_id in self._contexts:
            raise AgentGraphError(f"Agent '{node_id}' is already running")

        node = self._graph.get_node(node_id)
        session = await self.get_session(node_id)

        try:
            context = await session.__aenter__()
            self._contexts[node_id] = context
            node.status = NodeStatus.ACTIVE
            logger.info(f"Started agent '{node_id}' (status: {node.status.value})")

        except Exception as e:
            node.status = NodeStatus.ERROR
            node.metadata["startup_error"] = str(e)
            logger.error(f"Failed to start agent '{node_id}': {e}")
            raise AgentGraphError(f"Failed to start agent '{node_id}': {e}") from e

    async def stop_agent(self, node_id: str) -> None:
        """
        Stop an agent session gracefully.

        Exits the async context and updates status to STOPPED.
        Idempotent - can be called on non-running agents.

        Args:
            node_id: ID of the node to stop

        Raises:
            NodeNotFoundError: If node doesn't exist
            AgentGraphError: If stop operation fails
        """
        if node_id not in self._contexts:
            logger.warning(f"Agent '{node_id}' is not running - nothing to stop")
            return

        context = self._contexts.pop(node_id)
        node = self._graph.get_node(node_id)

        try:
            await context.__aexit__(None, None, None)
            node.status = NodeStatus.STOPPED
            logger.info(f"Stopped agent '{node_id}' (status: {node.status.value})")

        except Exception as e:
            logger.error(f"Error stopping agent '{node_id}': {e}")
            raise AgentGraphError(f"Error stopping agent '{node_id}': {e}") from e

    async def restart_agent(self, node_id: str) -> None:
        """
        Stop and restart an agent with fresh context.

        Args:
            node_id: ID of the node to restart

        Raises:
            NodeNotFoundError: If node doesn't exist
            AgentGraphError: If restart operation fails
        """
        logger.info(f"Restarting agent '{node_id}'...")
        await self.stop_agent(node_id)
        await asyncio.sleep(0.1)  # Brief pause between stop and start
        await self.start_agent(node_id)
        logger.info(f"Agent '{node_id}' restarted successfully")

    async def stop_all(self) -> None:
        """
        Stop all running agents (for cleanup).

        Errors stopping individual agents are logged but don't prevent
        stopping other agents.
        """
        node_ids = list(self._contexts.keys())
        logger.info(f"Stopping {len(node_ids)} running agents...")

        for node_id in node_ids:
            try:
                await self.stop_agent(node_id)
            except Exception as e:
                logger.error(f"Error stopping agent '{node_id}' during cleanup: {e}")

        logger.info("All agents stopped")

    async def with_retry(
        self,
        func: Callable[..., Any],
        node_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function with retry logic and exponential backoff.

        Updates Node.status to ERROR if all retries are exhausted.
        Clears error state on success.

        Args:
            func: Async function to execute
            node_id: ID of the node (for error tracking)
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            Result from func

        Raises:
            AgentGraphError: If all retries are exhausted
            NodeNotFoundError: If node doesn't exist
        """
        node = self._graph.get_node(node_id)

        for attempt in range(self.MAX_RETRIES):
            try:
                result = await func(*args, **kwargs)

                # Success - clear error state
                node.metadata.pop("error_count", None)
                node.metadata.pop("last_error", None)

                if attempt > 0:
                    logger.info(
                        f"Agent '{node_id}' recovered after {attempt} "
                        f"{'retry' if attempt == 1 else 'retries'}"
                    )

                return result

            except Exception as e:
                logger.warning(
                    f"Agent '{node_id}' error (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}"
                )

                if attempt == self.MAX_RETRIES - 1:
                    # Final failure - update error state
                    node.status = NodeStatus.ERROR
                    node.metadata["last_error"] = str(e)
                    node.metadata["error_count"] = node.metadata.get("error_count", 0) + 1

                    error_msg = (
                        f"Agent '{node_id}' failed after {self.MAX_RETRIES} retries. "
                        f"Total failures: {node.metadata['error_count']}"
                    )
                    logger.error(error_msg)
                    raise AgentGraphError(error_msg) from e

                # Wait before retry with exponential backoff
                delay = self.RETRY_DELAYS[attempt]
                logger.info(f"Retrying agent '{node_id}' in {delay}s...")
                await asyncio.sleep(delay)

        # Should never reach here, but for type safety
        raise AgentGraphError(f"Unexpected error in retry logic for agent '{node_id}'")

    async def start_agent_with_recovery(self, node_id: str) -> None:
        """
        Start an agent with error recovery and retry logic.

        Args:
            node_id: ID of the node to start

        Raises:
            NodeNotFoundError: If node doesn't exist
            AgentGraphError: If start fails after all retries
        """

        async def _start() -> None:
            await self.start_agent(node_id)

        await self.with_retry(_start, node_id)

    def is_running(self, node_id: str) -> bool:
        """
        Check if an agent is currently running.

        Args:
            node_id: ID of the node

        Returns:
            True if agent is running, False otherwise
        """
        return node_id in self._contexts

    def get_running_agents(self) -> list[str]:
        """
        Get list of all currently running agent node IDs.

        Returns:
            List of node IDs with active sessions
        """
        return list(self._contexts.keys())

    def get_session_count(self) -> int:
        """
        Get count of created sessions (running or stopped).

        Returns:
            Number of sessions created
        """
        return len(self._sessions)

    def get_running_count(self) -> int:
        """
        Get count of currently running agents.

        Returns:
            Number of agents with active sessions
        """
        return len(self._contexts)
