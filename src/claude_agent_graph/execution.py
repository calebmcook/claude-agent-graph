"""
Execution modes for agent graph orchestration.

Defines three execution modes:
- ManualController: Step-by-step manual control
- ReactiveExecutor: Message-driven automatic execution
- ProactiveExecutor: Periodic agent activation
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import AgentGraph

logger = logging.getLogger(__name__)


class ExecutionMode(ABC):
    """
    Abstract base class for execution modes.

    Execution modes control how messages are processed and agents are executed
    in the graph. Only one execution mode can be active at a time.
    """

    def __init__(self, graph: "AgentGraph") -> None:
        """
        Initialize an execution mode.

        Args:
            graph: Reference to the AgentGraph instance
        """
        self._graph = graph
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """
        Start the execution mode.

        Raises:
            RuntimeError: If execution mode is already running
        """
        if self._running:
            raise RuntimeError("Execution mode already running")

        self._running = True
        self._task = asyncio.create_task(self._execute_loop())
        logger.info(f"Started {self.__class__.__name__}")

    async def stop(self) -> None:
        """Stop the execution mode and wait for cleanup."""
        self._running = False
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped {self.__class__.__name__}")

    @abstractmethod
    async def _execute_loop(self) -> None:
        """
        Main execution loop.

        Subclasses must implement this method to define how messages
        are processed and agents are executed.
        """
        pass


class ManualController(ExecutionMode):
    """
    Manual execution mode for step-by-step control.

    The graph doesn't automatically process messages. The user must explicitly
    call step() to execute specific agents or step_all() to execute all agents
    with pending messages.
    """

    async def _execute_loop(self) -> None:
        """Manual controller doesn't have an automated loop."""
        # Simply wait until stopped
        while self._running:
            await asyncio.sleep(0.1)

    async def step(self, node_id: str) -> None:
        """
        Execute one turn for a specific agent.

        Processes one pending message from the agent's queue, if any.
        If no messages are pending, does nothing.

        Args:
            node_id: ID of the agent to execute

        Raises:
            ValueError: If node_id doesn't exist
        """
        if node_id not in self._graph._nodes:
            raise ValueError(f"Node '{node_id}' not found")

        # Get message queue for this node
        if node_id not in self._graph._message_queues:
            # No queue yet = no pending messages
            return

        queue = self._graph._message_queues[node_id]

        # Try to get one message without blocking
        try:
            message = queue.get_nowait()
            logger.debug(f"Processing message for node '{node_id}': {message.message_id}")

            # Process the message with the agent (calls _process_message_with_agent)
            response = await self._graph._process_message_with_agent(node_id, message)
            if response:
                logger.debug(f"Agent '{node_id}' response: {response[:100]}...")

        except asyncio.QueueEmpty:
            # No pending messages
            pass

    async def step_all(self) -> int:
        """
        Execute one turn for all agents with pending messages.

        Returns:
            Number of agents that had pending messages and were executed
        """
        executed_count = 0

        # Iterate over all nodes with pending messages
        for node_id in list(self._graph._message_queues.keys()):
            queue = self._graph._message_queues[node_id]

            # Check if queue has pending messages
            if not queue.empty():
                await self.step(node_id)
                executed_count += 1

        if executed_count > 0:
            logger.debug(f"Manual step_all: executed {executed_count} agents")

        return executed_count


class ReactiveExecutor(ExecutionMode):
    """
    Reactive execution mode (event-driven).

    Automatically processes messages from agent queues as they arrive.
    When a message is added to an agent's queue, the agent processes it
    asynchronously.
    """

    async def _execute_loop(self) -> None:
        """
        Main reactive loop.

        Monitors all message queues and processes messages as they arrive.
        """
        tasks = []

        try:
            while self._running:
                # Get all current queues
                queue_dict = self._graph._message_queues.copy()

                # Create monitor tasks for all queues
                monitor_tasks = [
                    asyncio.create_task(self._monitor_queue(node_id, queue))
                    for node_id, queue in queue_dict.items()
                ]

                if monitor_tasks:
                    try:
                        # Wait for any queue to have a message
                        done, pending = await asyncio.wait(
                            monitor_tasks,
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=1.0  # Check for new queues every second
                        )

                        # Cancel pending tasks
                        for task in pending:
                            task.cancel()

                        # Process completed tasks (queues with messages)
                        for task in done:
                            try:
                                node_id, message = await task
                                logger.debug(f"Reactive executor: processing {message.message_id} for '{node_id}'")

                                # Process the message with the agent (calls _process_message_with_agent)
                                response = await self._graph._process_message_with_agent(node_id, message)
                                if response:
                                    logger.debug(f"Agent '{node_id}' response: {response[:100]}...")

                            except asyncio.CancelledError:
                                pass
                            except Exception as e:
                                logger.error(f"Error processing message: {e}")
                    except asyncio.TimeoutError:
                        # Timeout just means no messages yet, continue
                        pass
                else:
                    # No queues yet, wait a bit before checking again
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            pass
        finally:
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _monitor_queue(self, node_id: str, queue: asyncio.Queue):
        """
        Monitor a queue for pending messages.

        Args:
            node_id: ID of the node
            queue: The message queue to monitor

        Returns:
            Tuple of (node_id, message) when a message is available
        """
        message = await queue.get()
        return node_id, message


class ProactiveExecutor(ExecutionMode):
    """
    Proactive execution mode (periodic agent activation).

    Agents wake up periodically and can initiate conversations without waiting
    for incoming messages. Useful for polling, monitoring, or background tasks.
    """

    def __init__(
        self,
        graph: "AgentGraph",
        interval: float = 60.0,
        start_delay: float = 0.0,
    ) -> None:
        """
        Initialize the proactive executor.

        Args:
            graph: Reference to the AgentGraph instance
            interval: Seconds between agent activations (default: 60)
            start_delay: Seconds before first activation (default: 0)
        """
        super().__init__(graph)
        self._interval = interval
        self._start_delay = start_delay

    async def _execute_loop(self) -> None:
        """
        Periodic agent activation loop.

        Sleeps for the configured interval, then activates each agent.
        """
        if self._start_delay > 0:
            await asyncio.sleep(self._start_delay)

        try:
            while self._running:
                # Activate each agent in the graph
                for node_id in self._graph.get_nodes():
                    if not self._running:
                        break

                    try:
                        logger.debug(f"Proactive activation of agent '{node_id}'")
                        # Start the agent session to allow it to be activated
                        await self._graph._agent_manager.start_agent(node_id)
                        logger.debug(f"Proactive executor: activated agent '{node_id}'")

                    except Exception as e:
                        logger.error(f"Error activating agent '{node_id}': {e}")

                # Sleep until next activation cycle
                await asyncio.sleep(self._interval)

        except asyncio.CancelledError:
            pass
