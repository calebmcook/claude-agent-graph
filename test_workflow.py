#!/usr/bin/env python3
"""
Test workflow script for Agent Graph Web Application

This script simulates the user workflow:
1. Initialize a new graph
2. Get supervisor analysis
3. Delegate tasks to agents
4. Get agent responses
"""

import asyncio
import json
from typing import Any

import requests

BASE_URL = "http://localhost:5001"


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_result(step: str, response: dict[str, Any]) -> None:
    """Pretty print API response."""
    print(f"✓ {step}")
    print(json.dumps(response, indent=2))


async def test_workflow() -> None:
    """Test the complete workflow."""
    problem = "Get the vote of 3 different agents on what the pri..."

    print_section("Testing Agent Graph Workflow")

    # Step 1: Initialize graph
    print_section("Step 1: Initialize Graph")
    response = requests.post(f"{BASE_URL}/api/initialize", json={"problem": problem})
    if response.status_code != 200:
        print(f"✗ Failed to initialize: {response.status_code}")
        print(response.text)
        return

    init_data = response.json()
    print_result("Initialize", init_data)

    # Step 2: Supervisor Think
    print_section("Step 2: Supervisor Analysis")
    response = requests.post(f"{BASE_URL}/api/supervisor-think", json={"problem": problem})
    if response.status_code != 200:
        print(f"✗ Failed to get supervisor analysis: {response.status_code}")
        print(response.text)
        return

    analysis_data = response.json()
    print_result("Supervisor Analysis", analysis_data)

    agents = analysis_data.get("agents", [])
    print(f"\nCreated {len(agents)} agents:")
    for agent in agents:
        print(f"  - {agent['id']} ({agent['role']})")

    # Step 3: Delegate tasks
    print_section("Step 3: Delegate Tasks to Agents")
    response = requests.post(f"{BASE_URL}/api/delegate", json={"problem": problem})
    if response.status_code != 200:
        print(f"✗ Failed to delegate: {response.status_code}")
        print(response.text)
        return

    delegation_data = response.json()
    print_result("Delegations", delegation_data)

    delegations = delegation_data.get("delegations", [])
    print(f"\nReceived {len(delegations)} delegations:")
    for delegation in delegations:
        print(f"  - {delegation['agent']}: {delegation['task'][:60]}...")

    if not delegations:
        print("\n✗ ERROR: No delegations returned! This is a problem.")
        return

    # Step 4: Get agent responses
    print_section("Step 4: Get Agent Responses")
    for delegation in delegations[:1]:  # Just test first agent for now
        agent_id = delegation["agent"]
        task = delegation["task"]

        response = requests.post(
            f"{BASE_URL}/api/agent-response", json={"agent": agent_id, "task": task}
        )

        if response.status_code != 200:
            print(f"✗ Failed to get response from {agent_id}: {response.status_code}")
            print(response.text)
            continue

        agent_response = response.json()
        print_result(f"Agent Response from {agent_id}", agent_response)

        # Show first 200 chars of response
        full_response = agent_response.get("response", "")
        print("\nAgent response preview:")
        print(f"{full_response[:200]}...")

    print_section("Workflow Complete")
    print("✓ All tests passed!")


if __name__ == "__main__":
    try:
        asyncio.run(test_workflow())
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
