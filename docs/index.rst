claude-agent-graph Documentation
=================================

**claude-agent-graph** is a Python package enabling creation and orchestration of
large-scale graphs where each node represents an independent Claude agent session.
Build complex, interconnected networks of AI agents that collaborate and maintain
shared state through structured conversation channels.

Features
--------

- 🕸️ **Flexible Graph Topologies**: Support for trees, DAGs, meshes, chains, stars, and cycles
- 🤖 **Independent Agent Sessions**: Each node is a full Claude agent via claude-agent-sdk
- 💬 **Shared Conversation State**: Persistent JSONL conversation files between agents
- 🎯 **Control Relationships**: Hierarchical authority via directed edges with automatic response routing
- ⚡ **Multiple Execution Modes**: Manual, reactive, and proactive orchestration
- 🔄 **Dynamic Operations**: Runtime graph modification with rollback support
- 💾 **Persistence & Recovery**: Checkpointing and crash recovery
- 🔌 **Pluggable Storage**: Filesystem, database, or custom backends
- 🎛️ **Per-Node Configuration**: Max tokens for cost/response control
- ✨ **Graceful Shutdown**: Clean agent termination with proper error handling

Quick Start
-----------

.. code-block:: python

    import asyncio
    from claude_agent_graph import AgentGraph
    from claude_agent_graph.backends import FilesystemBackend
    from claude_agent_graph.execution import ManualController

    async def main():
        async with AgentGraph(
            name="my_network",
            storage_backend=FilesystemBackend(base_dir="./conversations")
        ) as graph:
            # Add agents with optional max_tokens configuration
            await graph.add_node(
                "coordinator",
                "You coordinate tasks and delegate to workers.",
                model="claude-sonnet-4-20250514"
            )
            await graph.add_node(
                "worker",
                "You execute tasks assigned to you.",
                model="claude-sonnet-4-20250514",
                max_tokens=200  # Limit responses to 200 tokens for cost control
            )

            # Create directed edge (supervisor → worker)
            # Worker responses automatically route back to supervisor
            await graph.add_edge("coordinator", "worker", directed=True)

            # Create executor and step through agents
            executor = ManualController(graph)
            await executor.start()

            # Send message
            await graph.send_message(
                "coordinator",
                "worker",
                "Please analyze the data and provide a summary."
            )

            # Step the worker to process the message
            await executor.step("worker")

            # Get conversation (includes both request and response)
            messages = await graph.get_conversation("coordinator", "worker")
            for msg in messages:
                print(f"{msg.from_node} → {msg.to_node}: {msg.content[:100]}...")

            await executor.stop()

    asyncio.run(main())

Installation
------------

.. code-block:: bash

    pip install claude-agent-graph

Requirements:
- Python 3.10+
- ``anthropic`` package
- ``claude-agent-sdk``

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/concepts
   user_guide/topologies
   user_guide/messaging
   user_guide/execution_modes
   user_guide/storage
   user_guide/persistence
   user_guide/best_practices
   user_guide/troubleshooting
   user_guide/faq

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/graph
   api/models
   api/storage
   api/execution
   api/topology
   api/agent_manager
   api/exceptions

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   examples
   changelog
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
