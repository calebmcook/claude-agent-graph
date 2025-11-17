Core Concepts
=============

Understanding the fundamental concepts of claude-agent-graph.

Node (Agent)
------------

A **node** represents an independent Claude agent session. Each node has:

* **Node ID**: Unique identifier (e.g., ``"agent_001"``)
* **System Prompt**: Instructions defining agent behavior
* **Model**: Claude model to use (e.g., ``"claude-sonnet-4-20250514"``)
* **Metadata**: Custom attributes for your application
* **Status**: Current state (INITIALIZING, ACTIVE, STOPPED, ERROR)
* **Agent Session**: Active ClaudeSDKClient instance

Creating a Node
~~~~~~~~~~~~~~~

.. code-block:: python

    await graph.add_node(
        node_id="analyst",
        system_prompt="You are a data analyst specializing in financial metrics.",
        model="claude-sonnet-4-20250514",
        metadata={"department": "finance", "role": "senior"}
    )

The node becomes an active Claude agent that can:

* Maintain conversation context
* Execute tools and commands
* Respond to messages
* Collaborate with other agents

Edge (Connection)
-----------------

An **edge** represents a connection between two nodes. Each edge has:

* **From Node**: Source node ID
* **To Node**: Target node ID
* **Directed**: Whether the relationship has direction (default: ``True``)
* **Conversation File**: Shared JSONL file storing messages
* **Properties**: Custom metadata for the relationship

Creating an Edge
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Directed edge (one-way relationship)
    await graph.add_edge(
        "supervisor",
        "worker",
        directed=True,
        properties={"type": "reports_to", "priority": "high"}
    )

    # Undirected edge (bidirectional relationship)
    await graph.add_edge(
        "agent1",
        "agent2",
        directed=False
    )

Directed vs Undirected
~~~~~~~~~~~~~~~~~~~~~~

**Directed edges** create hierarchical relationships:

* Source node is a "controller" of target node
* Target node's system prompt is automatically modified to include controller info
* Enables ``execute_command()`` for control commands
* Models organizational hierarchies, workflows

**Undirected edges** create peer relationships:

* Both nodes can communicate freely
* No control authority
* Models collaborative networks, mesh topologies

Conversation Files
------------------

Each edge has an associated **conversation file** (``convo.jsonl``) that stores
the complete message history between two nodes.

Message Format
~~~~~~~~~~~~~~

Messages are stored as JSONL (newline-delimited JSON):

.. code-block:: json

    {"timestamp": "2025-11-04T12:00:00.000Z", "from_node": "agent1", "to_node": "agent2", "message_id": "msg_abc123", "role": "user", "content": "Hello", "metadata": {}}
    {"timestamp": "2025-11-04T12:00:01.500Z", "from_node": "agent2", "to_node": "agent1", "message_id": "msg_def456", "role": "assistant", "content": "Hi!", "metadata": {}}

Features:

* **Append-only**: Messages are never modified, only added
* **Thread-safe**: Concurrent writes are safe
* **Timestamped**: Microsecond precision for ordering
* **Filterable**: Query by timestamp or limit
* **Archivable**: Automatic log rotation when size exceeds limit

Reading Conversations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get all messages
    all_messages = await graph.get_conversation("agent1", "agent2")

    # Get recent messages
    recent = await graph.get_recent_messages("agent1", "agent2", count=10)

    # Get messages since a timestamp
    from datetime import datetime, timezone
    since = datetime(2025, 11, 4, tzinfo=timezone.utc)
    messages = await graph.get_conversation("agent1", "agent2", since=since)

    # Get limited messages
    messages = await graph.get_conversation("agent1", "agent2", limit=50)

Control Relationships
---------------------

When you create a **directed edge**, a control relationship is established:

.. code-block:: python

    await graph.add_edge("manager", "employee", directed=True)

This automatically:

1. Marks ``manager`` as a controller of ``employee``
2. Modifies ``employee``'s system prompt to include:

   .. code-block:: text

       Note: You are controlled by the following nodes: ['manager']
       You should follow directives from these controllers while maintaining your specialized role.

3. Enables ``execute_command()`` for authorized commands

Querying Control Relationships
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Who controls this node?
    controllers = graph.get_controllers("employee")
    # Returns: ['manager']

    # Who does this node control?
    subordinates = graph.get_subordinates("manager")
    # Returns: ['employee']

    # Check specific relationship
    is_controller = graph.is_controller("manager", "employee")
    # Returns: True

Executing Control Commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Controllers can issue commands to subordinates:

.. code-block:: python

    await graph.execute_command(
        controller="manager",
        subordinate="employee",
        command="generate_report",
        dataset="Q3_2025",
        format="pdf"
    )

This creates a special message with command metadata that the subordinate receives.

Topology
--------

The **topology** is the overall structure of your agent network. Common topologies:

* **Tree**: Hierarchical structure (e.g., org chart)
* **DAG**: Directed acyclic graph (e.g., workflow pipeline)
* **Chain**: Linear sequence (e.g., assembly line)
* **Star**: Central hub with spokes (e.g., dispatcher pattern)
* **Mesh**: Highly connected network (e.g., collaborative team)
* **Cycle**: Circular structure (e.g., iterative process)

Detecting Topology
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from claude_agent_graph.topology import GraphTopology

    topology = graph.get_topology()
    print(topology)  # e.g., GraphTopology.TREE

    # Validate required topology
    graph.validate_topology(GraphTopology.DAG)
    # Raises TopologyViolationError if not a DAG

See :doc:`topologies` for detailed topology guide.

Storage Backends
----------------

Storage backends handle conversation file persistence. The default is
``FilesystemBackend``, but you can implement custom backends.

.. code-block:: python

    from claude_agent_graph.backends import FilesystemBackend

    # Custom filesystem location
    backend = FilesystemBackend(
        base_dir="/data/conversations",
        max_size_mb=100.0  # Max file size before rotation
    )

    async with AgentGraph(name="my_graph", storage_backend=backend) as graph:
        # Your code here
        pass

See :doc:`storage` for more on storage backends.

Execution Modes
---------------

Execution modes control how agents process messages:

* **Manual**: Step-by-step control (call ``step()`` explicitly)
* **Reactive**: Automatic message-driven processing
* **Proactive**: Periodic agent activation

.. code-block:: python

    # Start reactive mode
    await graph.start(mode="reactive")

    # Agents now automatically process incoming messages
    await graph.send_message("agent1", "agent2", "Task")
    # agent2 processes automatically

    # Stop execution mode
    await graph.stop_execution()

See :doc:`execution_modes` for detailed execution guide.

Persistence & Checkpoints
--------------------------

Save and restore graph state:

.. code-block:: python

    # Save checkpoint
    await graph.save_checkpoint("./checkpoints/my_graph.ckpt")

    # Load checkpoint
    graph = await AgentGraph.load_checkpoint("./checkpoints/my_graph.ckpt")

    # Auto-save every 5 minutes
    graph = AgentGraph(
        name="my_graph",
        auto_save=True,
        auto_save_interval=300
    )
    await graph.start_auto_save()

See :doc:`persistence` for more on checkpointing.

Next Steps
----------

* :doc:`topologies` - Learn about graph structures
* :doc:`messaging` - Message routing patterns
* :doc:`execution_modes` - Control agent execution
* :doc:`best_practices` - Best practices and patterns
