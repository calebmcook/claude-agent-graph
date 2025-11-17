Quick Start
===========

This guide will get you up and running with claude-agent-graph in 5 minutes.

Create Your First Graph
------------------------

Here's a minimal example that creates two agents and has them communicate:

.. code-block:: python

    import asyncio
    from claude_agent_graph import AgentGraph

    async def main():
        # Create a graph with a name
        async with AgentGraph(name="quickstart_demo") as graph:

            # Add two agent nodes
            await graph.add_node(
                "coordinator",
                "You are a project coordinator. Delegate tasks to workers."
            )

            await graph.add_node(
                "worker",
                "You are a worker. Complete tasks assigned to you."
            )

            # Connect them with a directed edge (coordinator controls worker)
            await graph.add_edge("coordinator", "worker", directed=True)

            # Send a message from coordinator to worker
            await graph.send_message(
                "coordinator",
                "worker",
                "Please analyze the Q3 sales data and provide insights."
            )

            # Retrieve the conversation
            messages = await graph.get_conversation("coordinator", "worker")
            print(f"Conversation has {len(messages)} message(s)")

    if __name__ == "__main__":
        asyncio.run(main())

What's Happening?
-----------------

1. **Create a Graph**: ``AgentGraph(name="quickstart_demo")`` creates a new agent network.
   The ``async with`` context manager ensures proper cleanup.

2. **Add Nodes**: ``add_node()`` creates agent sessions with custom system prompts.
   Each node is an independent Claude agent.

3. **Add Edge**: ``add_edge()`` creates a connection between agents. The ``directed=True``
   parameter establishes a control relationship (coordinator controls worker).

4. **Send Message**: ``send_message()`` routes a message from one agent to another.
   The message is persisted to a conversation file.

5. **Get Conversation**: ``get_conversation()`` retrieves the conversation history
   between two agents.

Next Example: Hierarchy
------------------------

Create a simple organizational hierarchy:

.. code-block:: python

    async with AgentGraph(name="org_hierarchy") as graph:
        # Add managers and workers
        await graph.add_node("ceo", "You are the CEO.")
        await graph.add_node("vp_eng", "You are VP of Engineering.")
        await graph.add_node("engineer", "You are a software engineer.")

        # Create hierarchy
        await graph.add_edge("ceo", "vp_eng", directed=True)
        await graph.add_edge("vp_eng", "engineer", directed=True)

        # Message flows down the hierarchy
        await graph.send_message("ceo", "vp_eng", "Prioritize Q1 features")
        await graph.send_message("vp_eng", "engineer", "Work on auth module")

        # Check the hierarchy
        print(f"Engineer reports to: {graph.get_controllers('engineer')}")
        # Output: ['vp_eng']

Storage Location
----------------

By default, conversation files are stored in:

.. code-block:: text

    ./conversations/{graph_name}/
        edge_{node1}_{node2}/
            convo.jsonl

You can customize the storage location:

.. code-block:: python

    from claude_agent_graph import AgentGraph
    from claude_agent_graph.backends import FilesystemBackend

    async with AgentGraph(
        name="my_graph",
        storage_backend=FilesystemBackend(base_dir="/custom/path")
    ) as graph:
        # Your code here
        pass

Or use the convenience parameter:

.. code-block:: python

    async with AgentGraph(
        name="my_graph",
        storage_path="/custom/path"
    ) as graph:
        # Your code here
        pass

Running Examples
----------------

The package includes several example scripts in the ``examples/`` directory:

.. code-block:: bash

    # Simple tree hierarchy
    python examples/tree_hierarchy.py

    # DAG pipeline
    python examples/dag_pipeline.py

    # Star dispatcher pattern
    python examples/star_dispatcher.py

Next Steps
----------

* :doc:`concepts` - Understand core concepts in depth
* :doc:`topologies` - Learn about different graph structures
* :doc:`messaging` - Explore message routing patterns
* :doc:`execution_modes` - Control how agents execute
* :doc:`../examples` - More complex examples
