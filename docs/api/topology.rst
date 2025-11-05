Topology Utilities
==================

Graph topology detection and validation.

.. automodule:: claude_agent_graph.topology
   :members:
   :undoc-members:
   :show-inheritance:

GraphTopology Enum
------------------

.. autoclass:: claude_agent_graph.topology.GraphTopology
   :members:
   :undoc-members:

Topology Detection
------------------

.. autofunction:: claude_agent_graph.topology.detect_topology
.. autofunction:: claude_agent_graph.topology.validate_topology

Topology Checks
---------------

.. autofunction:: claude_agent_graph.topology.is_tree
.. autofunction:: claude_agent_graph.topology.is_dag
.. autofunction:: claude_agent_graph.topology.is_chain
.. autofunction:: claude_agent_graph.topology.is_star
.. autofunction:: claude_agent_graph.topology.is_cycle_graph
.. autofunction:: claude_agent_graph.topology.has_cycles
.. autofunction:: claude_agent_graph.topology.is_connected

Node Queries
------------

.. autofunction:: claude_agent_graph.topology.get_root_nodes
.. autofunction:: claude_agent_graph.topology.get_leaf_nodes
.. autofunction:: claude_agent_graph.topology.get_isolated_nodes
