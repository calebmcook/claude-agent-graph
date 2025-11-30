# Epic 8 Implementation Plan: Monitoring & Observability

**Status:** Ready for Implementation

**Date:** November 2025

**Estimated Effort:** 3-4 days

**Dependencies:** Epic 6 (Agent Execution & Control) ✅

---

## Overview

Epic 8 adds comprehensive observability and monitoring capabilities to the claude-agent-graph system. This enables users to track system health, monitor agent activity, analyze graph topology, and visualize agent networks. The implementation focuses on providing real-time metrics collection and flexible export formats for visualization.

**Key Capabilities:**
- Compute comprehensive metrics (node count, edge count, messages, utilization, error rates)
- Export graph topology as GraphViz (DOT) format for visualization tools
- Export graph topology as JSON for custom visualization and analysis
- Structured logging across all system components
- Performance tracking without impacting core operations
- Real-time system health monitoring

**Current State:**
The codebase already has logging integrated throughout (`logger.debug()`, `logger.info()`, `logger.error()` calls). Epic 8 builds on this foundation to add structured metrics collection and visualization export capabilities.

---

## Architecture Decisions

### 1. Lazy Metric Calculation Pattern

**Decision:** Implement metrics computation on-demand rather than maintaining real-time counters.

**Rationale:**
- Graph modification operations are relatively infrequent
- Metric queries are typically episodic (monitoring dashboards, debugging)
- Avoids overhead of counter updates on every `add_node`, `send_message`, etc.
- Leverages existing graph data structures directly
- Simplifies concurrency (no counter synchronization needed)
- For expensive metrics (message_count), provide optional caching with TTL

**Trade-offs:**
- First metrics query after large graph modifications may be slower
- Mitigated by optional caching for frequently accessed metrics
- Acceptable for most use cases (users don't query metrics on every operation)

---

### 2. Structured Logging Architecture

**Decision:** Use Python's standard `logging` module with optional JSON structured format.

**Rationale:**
- No new dependencies (logging is stdlib)
- Flexible configuration via `logging.conf` files
- Compatible with log aggregation platforms (ELK, Splunk, DataDog)
- Easy to enable/disable by level
- Structured logging enables automatic parsing and analysis
- Minimal performance impact when logging is disabled

**Implementation:**
```python
# Standard format (already used)
logger.info(f"Added node '{node_id}' to graph '{self.name}'")

# Structured logging option
logger.info("node_added", extra={
    "graph_id": self.name,
    "node_id": node_id,
    "node_type": "agent"
})
```

---

### 3. Visualization Format Selection

**Decision:** Support both GraphViz (DOT) and JSON formats, each optimized for different use cases.

**Rationale:**
- GraphViz/DOT: Standard format, integrates with dozens of visualization tools (Graphviz, Gephi, yEd)
- JSON: Universal format, enables custom visualization (D3.js, Cytoscape.js, custom tools)
- Different users have different tooling preferences
- Both export from same underlying graph data
- Minimal code duplication with shared helper methods

**Format Details:**
- **DOT Format:** Using networkx's built-in `write_dot()` or manual generation
- **JSON Format:** Node-link format (standard for D3.js/Cytoscape.js compatibility)

---

### 4. Metrics Caching Strategy

**Decision:** Implement optional time-bounded caching for expensive metrics.

**Rationale:**
- Some metrics require file I/O (message_count from conversation files)
- Repeated queries within short time windows don't need recomputation
- TTL ensures metrics remain reasonably fresh
- Optional to avoid complexity when caching not needed

**Implementation:**
```python
@dataclass
class CachedMetric:
    value: Any
    timestamp: datetime
    ttl: int  # seconds

    def is_valid(self) -> bool:
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age < self.ttl
```

---

### 5. Agent Utilization Tracking

**Decision:** Calculate utilization from message frequency rather than instrumented agent execution.

**Rationale:**
- Non-invasive: Uses existing message data, doesn't require agent modifications
- Meaningful metric: Message frequency accurately reflects agent activity
- Avoids overhead: No execution instrumentation needed
- Practical: Utilization computed as messages per time window
- Flexible: Window size configurable

**Calculation:**
- Utilization = (Messages sent by agent in window) / (Window duration in seconds)
- Window: Last 24 hours by default, configurable
- Result: Messages per hour or messages per day

---

### 6. Error Rate Calculation

**Decision:** Compute error rate from transaction logs and node status tracking.

**Rationale:**
- Transaction logs already capture operation failures (Epic 5)
- Node status already tracks ERROR state
- Non-invasive: Doesn't require agent-level error instrumentation
- Comprehensive: Captures graph operation failures and agent failures
- Time-windowed for trend analysis

**Calculation:**
- Error rate = (Failed operations) / (Total operations) in time window
- Failed operations from: transaction log failures + nodes in ERROR state
- Result: Percentage or ratio

---

### 7. Visualization Metadata Richness

**Decision:** Include rich node/edge attributes in visualization exports for context and filtering.

**Rationale:**
- Enables visual differentiation in external tools (color by status, size by activity)
- Supports filtering and analysis in visualization software
- Minimal export overhead (already have data)
- Enables advanced analysis (find error nodes, identify isolated components)
- Improves debugging experience

**Exported Metadata:**
- Nodes: node_id, status, model, metadata, created_at, message_count
- Edges: from_node, to_node, directed, properties, created_at, message_count
- Graph: name, node_count, edge_count, created_at

---

## Feature 8.1: Metrics Collection

### Story 8.1.1: Implement Graph Metrics

**Goal:** Provide comprehensive metrics about graph state and activity.

**Files to Create:**
- None (metrics integrated into graph.py)

**Files to Modify:**
- `src/claude_agent_graph/graph.py` (+200 lines)
- `tests/test_graph.py` (+100 lines)

**Implementation Details:**

```python
# Return type
@dataclass
class GraphMetrics:
    node_count: int
    edge_count: int
    message_count: int
    active_conversations: int
    avg_node_degree: float
    isolated_nodes: int
    agent_utilization: Dict[str, float]  # node_id -> messages/hour
    error_rate: float
    timestamp: datetime

# Method signature
def get_metrics(
    self,
    use_cache: bool = True,
    cache_ttl: int = 300,
    time_window: int = 86400  # 24 hours
) -> GraphMetrics:
    """
    Get comprehensive metrics about the graph.

    Args:
        use_cache: Whether to use cached metrics if available
        cache_ttl: Cache time-to-live in seconds (default 5 minutes)
        time_window: Time window for utilization/error rate in seconds

    Returns:
        GraphMetrics object with all metrics

    Raises:
        GraphMetricsError: If metric computation fails
    """
```

**Implementation Steps:**

1. Create `GraphMetrics` dataclass in models.py for structured return type
2. Implement `get_metrics()` in AgentGraph class
3. Implement helper methods:
   - `_count_nodes()` - return len(self._nodes)
   - `_count_edges()` - return len(self._edges)
   - `_count_messages()` - scan conversation files, aggregate message counts
   - `_count_active_conversations()` - edges with messages in time_window
   - `_calculate_node_degree()` - average number of connections per node
   - `_count_isolated_nodes()` - nodes with no edges
   - `_calculate_agent_utilization()` - messages per agent per hour
   - `_compute_error_rate()` - failed operations / total operations
4. Add caching layer using simple dict with timestamp validation
5. Implement error handling for file I/O in message counting
6. Add logging for metric computation

**Edge Cases:**
- Empty graph: All counts are 0, averages are 0
- Time window beyond graph age: Compute with available data
- Message count file I/O errors: Catch and log, return best effort
- Concurrent graph modifications during metric computation: Use snapshots
- Very large graphs: Message counting may be slow (ok, that's why we cache)
- Deleted conversation files: Handle gracefully, skip

**Design Notes:**
- Message counting is the most expensive operation
- Use caching aggressively for this metric
- Optional: Maintain lightweight message counters in memory (counter trade-off)
- Consider using transaction log for error_rate (already has operation history)

**Acceptance Criteria:**
- ✅ `get_metrics()` returns GraphMetrics with all fields populated
- ✅ Node count and edge count are accurate
- ✅ Message count correctly aggregates from all conversation files
- ✅ Active conversations counts edges with recent messages
- ✅ Agent utilization shows reasonable values (0 for inactive, >0 for active)
- ✅ Error rate correctly computed from failures
- ✅ Caching works correctly (returns cached value within TTL)
- ✅ All metrics handle edge cases (empty graph, missing files, etc.)
- ✅ Metrics computation doesn't block graph operations (snapshot-based)
- ✅ Unit tests cover each metric calculation
- ✅ Integration tests verify metrics with known graph structures
- ✅ Performance acceptable even for large graphs (with caching)

---

### Story 8.1.2: Implement Event Logging

**Goal:** Ensure comprehensive, structured event logging throughout the system.

**Files to Modify:**
- `src/claude_agent_graph/graph.py` (enhance existing logging)
- `src/claude_agent_graph/agent_manager.py` (enhance existing logging)
- `src/claude_agent_graph/execution.py` (enhance existing logging)
- `tests/test_graph.py` (+50 lines for logging tests)

**Current State:**
Logging is already integrated throughout. This story enhances it:
- Make logging more structured and consistent
- Ensure all critical operations are logged
- Add log level configuration
- Document logging strategy

**Implementation Details:**

```python
# Enhanced structured logging pattern
logger.info(
    "node_added",
    extra={
        "event": "node_added",
        "graph_id": self.name,
        "node_id": node_id,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
)

# Standard log levels to use:
# DEBUG - detailed operation steps
# INFO - graph structure changes, operation completions
# WARNING - recoverable issues (retry attempts)
# ERROR - operation failures
# CRITICAL - system-level failures
```

**Implementation Steps:**

1. Review all logging calls in codebase for consistency
2. Standardize log levels across modules
3. Enhance critical operation logging:
   - Node add/remove/update: INFO level with node_id and changes
   - Edge add/remove/update: INFO level with edge_id
   - Message send: DEBUG level with from/to/size
   - Graph state changes: INFO level with summary
   - Errors: ERROR level with full context and exception
4. Add structured logging configuration example
5. Document logging output format
6. Add tests that verify log output

**Edge Cases:**
- Logging during exceptions: Include exception context
- Sensitive data: Don't log user data or system prompts in detail
- High-frequency operations: Debug level to avoid spam
- Long-running operations: Log progress at interval points

**Acceptance Criteria:**
- ✅ All graph modification operations logged at INFO level
- ✅ Error conditions logged at ERROR level with context
- ✅ Structured logging format documented
- ✅ Log levels consistent across modules (no mixed patterns)
- ✅ Logging doesn't impact performance noticeably
- ✅ Tests verify log output for critical operations
- ✅ Logging can be configured via standard logging config
- ✅ No sensitive data (prompts, system config) in logs

---

## Feature 8.2: Visualization Export

### Story 8.2.1: Implement GraphViz Export

**Goal:** Export graph topology in GraphViz DOT format for visualization tools.

**Files to Create:**
- `src/claude_agent_graph/visualization.py` (~200 lines)
- `tests/test_visualization.py` (~150 lines)

**Files to Modify:**
- `src/claude_agent_graph/__init__.py` (+2 lines - export)
- `src/claude_agent_graph/graph.py` (+30 lines - add method)

**Implementation Details:**

```python
# visualization.py

def export_graphviz(
    graph: "AgentGraph",
    output_path: Optional[Path | str] = None,
    include_metadata: bool = True,
    include_message_counts: bool = False,
    node_style: str = "default",  # "default", "status", "utilization"
    edge_style: str = "default"   # "default", "messages", "control"
) -> str:
    """
    Export graph as GraphViz DOT format.

    Args:
        graph: AgentGraph instance to export
        output_path: Path to write DOT file (optional, returns string if None)
        include_metadata: Include node/edge properties in labels
        include_message_counts: Add message count annotations
        node_style: How to style nodes ("default", "status", "utilization")
        edge_style: How to style edges ("default", "messages", "control")

    Returns:
        DOT format string

    Raises:
        VisualizationError: If export fails
    """
    # Returns something like:
    """
    digraph {
        node1 [label="node1\n(Active)", fillcolor=green, style=filled]
        node2 [label="node2\n(Active)", fillcolor=green, style=filled]
        node1 -> node2 [label="5 messages"]
    }
    """
```

**Implementation Steps:**

1. Create visualization.py module with export functions
2. Implement DOT generation:
   - Create digraph (directed graph)
   - Add nodes with IDs, labels, and optional styling
   - Add edges with direction indicators
   - Use node status for coloring (green=ACTIVE, red=ERROR, etc.)
   - Optional: Use node degree for size, message count for edge width
3. Optional: Use networkx.drawing.nx_agraph.to_agraph() for generation
4. Handle metadata formatting (truncate long values, escape special chars)
5. Implement style variants (status-based, utilization-based, etc.)
6. Write to file if output_path provided, else return string
7. Add tests with sample graphs

**Edge Cases:**
- Special characters in node_ids: Escape for DOT format
- Very large graphs: Still generate (may be slow but valid)
- Node statuses need color mapping: Define consistent palette
- Message count annotations: Truncate or summarize if too many
- Isolated nodes: Still include in visualization

**Design Notes:**
- Use networkx.drawing.nx_agraph if available, fallback to manual generation
- DOT format is simple enough for manual string building
- Color palette: ACTIVE=green, ERROR=red, STOPPED=gray, INITIALIZING=yellow
- Edge direction shown with arrows (-> for directed)
- Labels should be readable but not overwhelming

**Acceptance Criteria:**
- ✅ Export generates valid DOT format (can be parsed by graphviz tools)
- ✅ All nodes included with proper IDs and labels
- ✅ All edges included with direction indicators
- ✅ Optional metadata displayed in node/edge labels
- ✅ Optional message count annotations on edges
- ✅ Node styling reflects status (colors match status enum)
- ✅ Export to file or string both work
- ✅ Large graphs export without errors
- ✅ Special characters properly escaped
- ✅ Generated DOT renders correctly in graphviz, Gephi, yEd
- ✅ Unit tests verify DOT syntax
- ✅ Integration tests with various graph structures

---

### Story 8.2.2: Implement JSON Export

**Goal:** Export graph topology in JSON format for custom visualization tools.

**Files to Modify:**
- `src/claude_agent_graph/visualization.py` (+100 lines)
- `tests/test_visualization.py` (+100 lines)

**Implementation Details:**

```python
# visualization.py

def export_json(
    graph: "AgentGraph",
    output_path: Optional[Path | str] = None,
    format: str = "node-link",  # "node-link" or "adjacency"
    include_metadata: bool = True,
    include_message_counts: bool = False
) -> Dict[str, Any] | str:
    """
    Export graph as JSON (node-link or adjacency format).

    Args:
        graph: AgentGraph instance to export
        output_path: Path to write JSON file (optional, returns dict if None)
        format: "node-link" (D3.js/Cytoscape compatible) or "adjacency"
        include_metadata: Include full node/edge properties
        include_message_counts: Include message statistics

    Returns:
        Dict with JSON structure (or JSON string if output_path provided)

    Raises:
        VisualizationError: If export fails
    """
    # Node-link format (standard for D3.js, Cytoscape.js):
    """
    {
        "nodes": [
            {"id": "node1", "status": "active", "model": "...", "metadata": {...}},
            {"id": "node2", "status": "active", "model": "...", "metadata": {...}}
        ],
        "links": [
            {"source": "node1", "target": "node2", "directed": true, "properties": {...}}
        ]
    }
    """
```

**Implementation Steps:**

1. Implement node-link format exporter:
   - Extract nodes list with id, status, model, metadata
   - Extract links list (edges) with source, target, directed flag, properties
   - Include edge metadata (created_at, properties, message_count if requested)
   - Ensure all datetime objects converted to ISO strings
2. Implement adjacency format exporter (alternative):
   - Nodes as flat list with all attributes
   - Adjacency dict mapping node_id to list of neighbor_ids
3. Add message_count calculation if requested
4. Handle datetime serialization (convert to ISO 8601 strings)
5. Write to file if output_path provided, else return dict
6. Return JSON string if output_path provided, dict otherwise
7. Tests verify JSON structure and compatibility

**Edge Cases:**
- Datetime objects: Convert to ISO 8601 strings for JSON compatibility
- Circular references: Node-link format naturally avoids these
- None values: Include in JSON or exclude? (Include for completeness)
- Very large graphs: JSON serialization still efficient
- Special characters in node_ids: JSON encoding handles this

**Design Notes:**
- Node-link format is standard (widely supported by D3.js, Cytoscape.js, etc.)
- Use json.dumps with indent for readability
- Include timestamps for all objects (enables sorting/filtering)
- Message counts computed at export time (use metrics if requested)
- Metadata structure preserved exactly as in Python

**Acceptance Criteria:**
- ✅ Exports valid JSON that can be parsed
- ✅ All nodes included with complete properties
- ✅ All edges included with source/target/directed/properties
- ✅ Datetime objects converted to ISO 8601 strings
- ✅ Node-link format compatible with D3.js and Cytoscape.js
- ✅ Optional metadata included/excluded as requested
- ✅ Message counts accurate (if included)
- ✅ Export to file or dict both work
- ✅ Large graphs export without errors
- ✅ JSON can be imported into visualization tools
- ✅ Unit tests verify JSON structure
- ✅ Integration tests verify compatibility with standard tools

---

## Testing Strategy

### Unit Tests

**File:** `tests/test_visualization.py` (~220 lines)

**Test Categories:**

1. **GraphViz Export** (7 tests)
   - `test_export_graphviz_basic_graph` - simple node/edge structure
   - `test_export_graphviz_with_metadata` - includes properties
   - `test_export_graphviz_isolated_nodes` - single nodes no edges
   - `test_export_graphviz_with_special_chars` - escaping in node ids
   - `test_export_graphviz_color_by_status` - status styling
   - `test_export_graphviz_to_file` - file output works
   - `test_export_graphviz_syntax_valid` - output is valid DOT

2. **JSON Export** (6 tests)
   - `test_export_json_node_link_format` - basic node-link structure
   - `test_export_json_preserves_metadata` - all properties included
   - `test_export_json_datetime_encoding` - ISO 8601 timestamps
   - `test_export_json_with_message_counts` - message stats included
   - `test_export_json_to_file` - file output works
   - `test_export_json_parsing` - output can be reparsed

**File:** Updated `tests/test_graph.py` (+50 lines)

3. **Metrics** (8 tests)
   - `test_metrics_node_count` - accurate node counting
   - `test_metrics_edge_count` - accurate edge counting
   - `test_metrics_message_count` - correct aggregation
   - `test_metrics_empty_graph` - handles zero state
   - `test_metrics_caching` - cache TTL works
   - `test_metrics_agent_utilization` - reasonable values
   - `test_metrics_error_rate` - computed correctly
   - `test_metrics_large_graph` - performance acceptable

4. **Event Logging** (5 tests)
   - `test_logging_node_operations` - add/remove logged
   - `test_logging_edge_operations` - add/remove logged
   - `test_logging_errors` - exceptions logged at ERROR level
   - `test_logging_structured_format` - consistent format
   - `test_logging_levels_appropriate` - DEBUG vs INFO usage

### Integration Tests

**In test_visualization.py and test_graph.py** (~30 lines)

1. `test_visualization_roundtrip` - export and analyze structure
2. `test_metrics_with_active_graph` - metrics accurate during operations
3. `test_visualization_compatibility` - exported JSON compatible with standard tools

### Coverage Target

- **>85% for new code** (visualization.py, metrics methods)
- **All 550+ existing tests must still pass**
- **New tests: ~28 total**

---

## Files Summary

### New Files

1. **`src/claude_agent_graph/visualization.py`** (~230 lines)
   - `export_graphviz()` function
   - `export_json()` function
   - Helper functions for formatting and styling
   - Type hints and comprehensive docstrings

2. **`tests/test_visualization.py`** (~220 lines)
   - Test class: `TestGraphVizExport` (7 tests)
   - Test class: `TestJSONExport` (6 tests)
   - Fixtures for sample graphs
   - Validation helpers

### Modified Files

1. **`src/claude_agent_graph/graph.py`** (+200 lines)
   - `get_metrics()` method
   - Helper methods for metric calculation
   - Cache management
   - Import GraphMetrics from models

2. **`src/claude_agent_graph/models.py`** (+30 lines)
   - `GraphMetrics` dataclass
   - Optional `CachedMetric` dataclass for caching

3. **`src/claude_agent_graph/__init__.py`** (+3 lines)
   - Export `GraphMetrics`
   - Export visualization functions

4. **`tests/test_graph.py`** (+100 lines)
   - Tests for `get_metrics()`
   - Tests for logging output
   - Integration tests

---

## Success Criteria

### Code Quality
- ✅ Test coverage >85% for new code (visualization, metrics)
- ✅ All 550+ existing tests pass (no regressions)
- ✅ Comprehensive type hints throughout
- ✅ Docstrings for all public methods
- ✅ Logging consistent and well-structured

### Functionality
- ✅ Metrics accurately reflect graph state
- ✅ GraphViz export produces valid DOT format
- ✅ JSON export compatible with D3.js/Cytoscape.js
- ✅ Caching works correctly with TTL
- ✅ All log messages appropriately leveled

### Performance
- ✅ Metrics computation with caching <500ms for typical graphs
- ✅ Visualization export <1s for graphs with <1000 nodes
- ✅ Logging adds <5% overhead
- ✅ No blocking of graph operations during metric computation

### Documentation
- ✅ Method docstrings include usage examples
- ✅ README updated with metrics and visualization examples
- ✅ Logging output format documented

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Message file counting slow on large graphs | Medium | Medium | Implement caching with TTL, consider in-memory counters |
| JSON datetime serialization issues | Low | Low | Use datetime.isoformat() which is standard |
| GraphViz special character escaping | Medium | Low | Use networkx if available, or implement robust escaping |
| Metrics computation during concurrent modifications | Medium | Medium | Use snapshot of graph state before computation |
| Visualization export file I/O errors | Low | Low | Proper error handling and informative error messages |
| Performance regression in core graph operations | Low | High | Don't add overhead to hot paths, metrics on-demand only |

---

## Implementation Timeline

### Phase 1: Metrics Collection (Day 1)
- Story 8.1.1: Implement Graph Metrics
  - Create GraphMetrics dataclass
  - Implement get_metrics() method
  - Implement all helper methods
  - Unit tests for metrics

### Phase 2: Visualization Exports (Day 2)
- Story 8.2.1: Implement GraphViz Export
  - Create visualization.py module
  - Implement export_graphviz()
  - Unit tests for DOT generation
- Story 8.2.2: Implement JSON Export
  - Implement export_json()
  - Unit tests for JSON structure

### Phase 3: Logging & Integration (Day 3)
- Story 8.1.2: Implement Event Logging
  - Enhance existing logging
  - Add structured logging tests
  - Integration tests for all features
  - Documentation and examples

**Estimated Total:** 3-4 days

**Complexity:** Medium (no major architectural changes, builds on existing systems)

---

## Future Enhancements

The following features are explicitly out of scope for Epic 8:
- Real-time metrics dashboard (Phase 5+)
- Metrics persistence to time-series database (Phase 5+)
- Advanced visualization with interactive filtering (Phase 5+)
- Performance profiling and tracing (Phase 5+)
- Custom metrics framework (Phase 5+)
- Prometheus metrics export (Phase 5+)

---

## References

- **IMPLEMENTATION_PLAN.md** - Lines 943-1020 (Epic 8 overview)
- **EPIC_6_IMPLEMENTATION.md** - Execution modes and message routing patterns
- **EPIC_5_IMPLEMENTATION.md** - Transaction logging framework

---

**Document Status:** Ready for Implementation

**Last Updated:** November 2025

**Author:** Implementation Team
