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
- 🎯 **Control Relationships**: Hierarchical authority via directed edges
- ⚡ **Multiple Execution Modes**: Manual, reactive, and proactive orchestration
- 🔄 **Dynamic Operations**: Runtime graph modification with rollback support
- 💾 **Persistence & Recovery**: Checkpointing and crash recovery
- 🔌 **Pluggable Storage**: Filesystem, database, or custom backends

Quick Start
-----------

.. code-block:: python

    from claude_agent_graph import AgentGraph

    async with AgentGraph(name="my_network") as graph:
        # Add agents
        await graph.add_node("agent1", "You are a coordinator.")
        await graph.add_node("agent2", "You are a worker.")

        # Connect them
        await graph.add_edge("agent1", "agent2", directed=True)

        # Send a message
        await graph.send_message("agent1", "agent2", "Process this task")

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
