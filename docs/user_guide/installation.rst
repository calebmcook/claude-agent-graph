Installation
============

Requirements
------------

* Python 3.10 or higher
* ``pip`` package manager
* Anthropic API key

Basic Installation
------------------

Install from PyPI using pip:

.. code-block:: bash

    pip install claude-agent-graph

This will install claude-agent-graph and its dependencies:

* ``anthropic`` - Claude API client
* ``claude-agent-sdk`` - Agent session management
* ``aiofiles`` - Async file I/O
* ``networkx`` - Graph algorithms
* ``pydantic`` - Data validation
* ``msgpack`` - Binary serialization

Development Installation
------------------------

For development or to use the latest features:

.. code-block:: bash

    git clone https://github.com/yourusername/claude-agent-graph.git
    cd claude-agent-graph
    pip install -e .
    pip install -r requirements-dev.txt

The development dependencies include:

* ``pytest`` - Testing framework
* ``pytest-asyncio`` - Async test support
* ``pytest-cov`` - Coverage reporting
* ``black`` - Code formatting
* ``ruff`` - Linting
* ``mypy`` - Type checking

Environment Setup
-----------------

Set your Anthropic API key as an environment variable:

.. code-block:: bash

    export ANTHROPIC_API_KEY=your_api_key_here

Or in Python:

.. code-block:: python

    import os
    os.environ["ANTHROPIC_API_KEY"] = "your_api_key_here"

Verification
------------

Verify your installation:

.. code-block:: python

    import claude_agent_graph
    print(claude_agent_graph.__version__)

Run the test suite:

.. code-block:: bash

    pytest tests/

Next Steps
----------

* :doc:`quickstart` - Get started with a simple example
* :doc:`concepts` - Learn core concepts
* :doc:`../examples` - Explore example scripts
