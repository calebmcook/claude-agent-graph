AgentGraph API
==============

The ``AgentGraph`` class is the main orchestration class for managing agent networks.
It handles graph construction, topology validation, message routing, and agent lifecycle management.

.. automodule:: claude_agent_graph.graph
   :members:
   :undoc-members:
   :show-inheritance:

Core Methods
------------

Graph Construction
~~~~~~~~~~~~~~~~~~

.. automethod:: claude_agent_graph.graph.AgentGraph.add_node
.. automethod:: claude_agent_graph.graph.AgentGraph.add_edge
.. automethod:: claude_agent_graph.graph.AgentGraph.remove_node
.. automethod:: claude_agent_graph.graph.AgentGraph.remove_edge
.. automethod:: claude_agent_graph.graph.AgentGraph.update_node
.. automethod:: claude_agent_graph.graph.AgentGraph.update_edge

Graph Queries
~~~~~~~~~~~~~

.. automethod:: claude_agent_graph.graph.AgentGraph.get_node
.. automethod:: claude_agent_graph.graph.AgentGraph.get_nodes
.. automethod:: claude_agent_graph.graph.AgentGraph.get_edge
.. automethod:: claude_agent_graph.graph.AgentGraph.get_edges
.. automethod:: claude_agent_graph.graph.AgentGraph.get_neighbors
.. automethod:: claude_agent_graph.graph.AgentGraph.node_exists
.. automethod:: claude_agent_graph.graph.AgentGraph.edge_exists

Message Routing
~~~~~~~~~~~~~~~

.. automethod:: claude_agent_graph.graph.AgentGraph.send_message
.. automethod:: claude_agent_graph.graph.AgentGraph.broadcast
.. automethod:: claude_agent_graph.graph.AgentGraph.route_message
.. automethod:: claude_agent_graph.graph.AgentGraph.get_conversation
.. automethod:: claude_agent_graph.graph.AgentGraph.get_recent_messages

Control Relationships
~~~~~~~~~~~~~~~~~~~~~

.. automethod:: claude_agent_graph.graph.AgentGraph.get_controllers
.. automethod:: claude_agent_graph.graph.AgentGraph.get_subordinates
.. automethod:: claude_agent_graph.graph.AgentGraph.is_controller
.. automethod:: claude_agent_graph.graph.AgentGraph.execute_command

Topology
~~~~~~~~

.. automethod:: claude_agent_graph.graph.AgentGraph.get_topology
.. automethod:: claude_agent_graph.graph.AgentGraph.validate_topology
.. automethod:: claude_agent_graph.graph.AgentGraph.get_isolated_nodes

Execution
~~~~~~~~~

.. automethod:: claude_agent_graph.graph.AgentGraph.start
.. automethod:: claude_agent_graph.graph.AgentGraph.stop_execution

Persistence
~~~~~~~~~~~

.. automethod:: claude_agent_graph.graph.AgentGraph.save_checkpoint
.. automethod:: claude_agent_graph.graph.AgentGraph.load_checkpoint
.. automethod:: claude_agent_graph.graph.AgentGraph.start_auto_save
.. automethod:: claude_agent_graph.graph.AgentGraph.stop_auto_save

Agent Lifecycle
~~~~~~~~~~~~~~~

.. automethod:: claude_agent_graph.graph.AgentGraph.start_agent
.. automethod:: claude_agent_graph.graph.AgentGraph.stop_agent
.. automethod:: claude_agent_graph.graph.AgentGraph.restart_agent
.. automethod:: claude_agent_graph.graph.AgentGraph.get_agent_status

Properties
----------

.. autoproperty:: claude_agent_graph.graph.AgentGraph.node_count
.. autoproperty:: claude_agent_graph.graph.AgentGraph.edge_count
