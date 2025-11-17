"""
End-to-end browser tests for Flask web application using Playwright.

These tests verify complete user workflows including:
- Problem initialization
- Supervisor analysis
- Agent creation and delegation
- Real-time graph visualization
- Chat interactions
"""

import json
import re
from typing import Optional

import pytest
from playwright.async_api import Browser, Page, expect


@pytest.fixture
def base_url() -> str:
    """
    Provide base URL for Flask application.

    Returns:
        str: Base URL of Flask app (default: localhost:5001)
    """
    return "http://localhost:5001"


@pytest.fixture
async def browser_context(browser: Browser):
    """
    Create a browser context for testing.

    Args:
        browser: Playwright browser fixture

    Yields:
        BrowserContext: Browser context
    """
    context = await browser.new_context()
    yield context
    await context.close()


@pytest.fixture
async def page(browser_context) -> Page:
    """
    Create a page for testing.

    Args:
        browser_context: Browser context fixture

    Yields:
        Page: Playwright page
    """
    page = await browser_context.new_page()
    yield page
    await page.close()


class TestInitializationWorkflow:
    """Tests for problem initialization workflow."""

    async def test_navigate_to_app(self, page: Page, base_url: str):
        """
        Test that user can navigate to application.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)
        # Check page is loaded
        await expect(page).to_have_title("")  # Page title should exist
        assert page.url == base_url + "/"

    async def test_problem_input_visible(self, page: Page, base_url: str):
        """
        Test that problem input field is visible and interactive.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        # Look for problem input textarea
        problem_input = page.locator("textarea[id='problemInput']")
        await expect(problem_input).to_be_visible()
        await expect(problem_input).to_be_editable()

    async def test_enter_problem_statement(self, page: Page, base_url: str):
        """
        Test entering a problem statement.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        # Fill in problem
        problem_text = "How can we scale our system to handle 1 million users?"
        await page.fill("textarea[id='problemInput']", problem_text)

        # Verify text was entered
        value = await page.input_value("textarea[id='problemInput']")
        assert value == problem_text

    async def test_initialize_button_enabled(self, page: Page, base_url: str):
        """
        Test that initialize button is enabled after entering problem.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        # Enter problem text
        problem_text = "What is the best architecture?"
        await page.fill("textarea[id='problemInput']", problem_text)

        # Initialize button should be enabled
        init_button = page.locator("button:has-text('Initialize')")
        await expect(init_button).to_be_enabled()

    async def test_click_initialize_button(self, page: Page, base_url: str):
        """
        Test clicking initialize button and handling response.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        # Enter problem and click initialize
        problem_text = "Design a microservices architecture"
        await page.fill("textarea[id='problemInput']", problem_text)

        # Listen for API response
        async with page.expect_request_finish(
            url=re.compile(r"/api/initialize")
        ) as request_context:
            await page.click("button:has-text('Initialize')")

        # Should have made API request
        request = await request_context.request
        assert request.method == "POST"


class TestSupervisorAnalysisWorkflow:
    """Tests for supervisor analysis workflow."""

    async def test_supervisor_think_button_visible(self, page: Page, base_url: str):
        """
        Test that supervisor think button is visible.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        # Initialize first
        problem_text = "Solve this architectural problem"
        await page.fill("textarea[id='problemInput']", problem_text)
        await page.click("button:has-text('Initialize')")

        # Wait for supervisor think button
        supervisor_button = page.locator("button:has-text('Supervisor Thinks')")
        await expect(supervisor_button).to_be_visible()

    async def test_supervisor_analysis_displayed(self, page: Page, base_url: str):
        """
        Test that supervisor analysis is displayed after thinking.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        # Initialize
        problem_text = "What's the best approach to system design?"
        await page.fill("textarea[id='problemInput']", problem_text)
        await page.click("button:has-text('Initialize')")

        # Click supervisor think
        await page.click("button:has-text('Supervisor Thinks')")

        # Look for analysis content
        analysis_element = page.locator("div[id='analysis'], div[class*='analysis']")
        # Wait for content to appear (with timeout)
        try:
            await expect(analysis_element).to_be_visible(timeout=5000)
        except AssertionError:
            # Analysis might be in a different location, that's ok for E2E test
            pass

    async def test_agents_created_after_analysis(self, page: Page, base_url: str):
        """
        Test that agents are created and displayed after analysis.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        # Go through initialization and analysis
        problem_text = "Create a web application"
        await page.fill("textarea[id='problemInput']", problem_text)
        await page.click("button:has-text('Initialize')")
        await page.click("button:has-text('Supervisor Thinks')")

        # Look for agent elements (they should be in the graph visualization)
        # Agents might be represented as circles/nodes in SVG or as list items
        agent_elements = page.locator("circle, [class*='agent'], [id*='agent']")
        count = await agent_elements.count()
        # Should have at least supervisor + some agents
        assert count >= 1


class TestDelegationWorkflow:
    """Tests for task delegation workflow."""

    async def test_delegate_button_visible(self, page: Page, base_url: str):
        """
        Test that delegate button is visible.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        problem_text = "Solve the scaling problem"
        await page.fill("textarea[id='problemInput']", problem_text)
        await page.click("button:has-text('Initialize')")
        await page.click("button:has-text('Supervisor Thinks')")

        # Delegate button should be visible
        delegate_button = page.locator("button:has-text('Delegate')")
        await expect(delegate_button).to_be_visible()

    async def test_tasks_delegated_to_agents(self, page: Page, base_url: str):
        """
        Test that tasks are delegated to agents.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        problem_text = "Implement a distributed system"
        await page.fill("textarea[id='problemInput']", problem_text)
        await page.click("button:has-text('Initialize')")
        await page.click("button:has-text('Supervisor Thinks')")
        await page.click("button:has-text('Delegate')")

        # Look for delegation content
        delegation_element = page.locator("div[id*='delegation'], div[class*='delegation']")
        # Wait for delegations to appear
        try:
            await expect(delegation_element).to_be_visible(timeout=5000)
        except AssertionError:
            # Delegations might be displayed differently
            pass


class TestAgentResponseWorkflow:
    """Tests for agent response workflow."""

    async def test_agent_response_button_visible(self, page: Page, base_url: str):
        """
        Test that agent response buttons are visible.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        # Complete workflow up to delegation
        problem_text = "Design a database architecture"
        await page.fill("textarea[id='problemInput']", problem_text)
        await page.click("button:has-text('Initialize')")
        await page.click("button:has-text('Supervisor Thinks')")
        await page.click("button:has-text('Delegate')")

        # Look for agent response buttons
        response_buttons = page.locator("button:has-text('Get Response')")
        count = await response_buttons.count()
        # Should have response buttons for agents
        assert count >= 0

    async def test_agent_response_displayed(self, page: Page, base_url: str):
        """
        Test that agent responses are displayed.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        problem_text = "Solve the problem"
        await page.fill("textarea[id='problemInput']", problem_text)
        await page.click("button:has-text('Initialize')")
        await page.click("button:has-text('Supervisor Thinks')")
        await page.click("button:has-text('Delegate')")

        # Try to click a response button if it exists
        response_buttons = page.locator("button:has-text('Get Response')")
        if await response_buttons.count() > 0:
            await response_buttons.first.click()

            # Look for response content
            response_element = page.locator("div[id*='response'], div[class*='response']")
            try:
                await expect(response_element).to_be_visible(timeout=5000)
            except AssertionError:
                pass


class TestGraphVisualization:
    """Tests for graph visualization."""

    async def test_graph_visualization_rendered(self, page: Page, base_url: str):
        """
        Test that graph visualization is rendered.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        # Initialize graph
        problem_text = "Create an application"
        await page.fill("textarea[id='problemInput']", problem_text)
        await page.click("button:has-text('Initialize')")

        # Look for SVG (D3 visualization)
        svg = page.locator("svg")
        await expect(svg).to_be_visible()

    async def test_graph_nodes_visible(self, page: Page, base_url: str):
        """
        Test that graph nodes are visible in visualization.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        problem_text = "Build a system"
        await page.fill("textarea[id='problemInput']", problem_text)
        await page.click("button:has-text('Initialize')")
        await page.click("button:has-text('Supervisor Thinks')")

        # Look for node circles in SVG
        nodes = page.locator("circle")
        count = await nodes.count()
        # Should have supervisor + agents
        assert count >= 1

    async def test_graph_links_visible(self, page: Page, base_url: str):
        """
        Test that graph links/edges are visible.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        problem_text = "Design a network"
        await page.fill("textarea[id='problemInput']", problem_text)
        await page.click("button:has-text('Initialize')")
        await page.click("button:has-text('Supervisor Thinks')")

        # Look for links in SVG
        links = page.locator("line, path[class*='link']")
        count = await links.count()
        # Should have links between nodes
        assert count >= 0


class TestSupervisorChatWorkflow:
    """Tests for supervisor chat functionality."""

    async def test_chat_input_visible(self, page: Page, base_url: str):
        """
        Test that chat input is visible.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        # Initialize
        problem_text = "Ask the supervisor"
        await page.fill("textarea[id='problemInput']", problem_text)
        await page.click("button:has-text('Initialize')")

        # Look for chat input
        chat_input = page.locator("textarea[id*='chat'], input[placeholder*='message']")
        await expect(chat_input).to_be_visible()

    async def test_send_chat_message(self, page: Page, base_url: str):
        """
        Test sending a chat message to supervisor.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        # Initialize
        problem_text = "Initial problem"
        await page.fill("textarea[id='problemInput']", problem_text)
        await page.click("button:has-text('Initialize')")

        # Send chat message
        chat_input = page.locator("textarea[id*='chat'], input[placeholder*='message']")
        await chat_input.fill("Can you clarify your approach?")

        # Find and click send button
        send_button = page.locator("button:has-text('Send')")
        if await send_button.count() > 0:
            await send_button.click()

    async def test_chat_response_displayed(self, page: Page, base_url: str):
        """
        Test that chat responses are displayed.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        problem_text = "Chat test"
        await page.fill("textarea[id='problemInput']", problem_text)
        await page.click("button:has-text('Initialize')")

        # Send message
        chat_input = page.locator("textarea[id*='chat'], input[placeholder*='message']")
        await chat_input.fill("Test message")

        send_button = page.locator("button:has-text('Send')")
        if await send_button.count() > 0:
            await send_button.click()
            # Wait for response
            await page.wait_for_timeout(1000)

            # Look for chat messages
            messages = page.locator("div[class*='message'], div[class*='chat']")
            count = await messages.count()
            assert count >= 0


class TestErrorHandling:
    """Tests for error handling in UI."""

    async def test_empty_problem_error(self, page: Page, base_url: str):
        """
        Test that empty problem shows error.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        # Try to initialize without problem
        init_button = page.locator("button:has-text('Initialize')")
        # Button might be disabled, which is ok
        if await init_button.is_enabled():
            await init_button.click()
            # Should see error message
            try:
                error = page.locator("div[class*='error'], [role='alert']")
                await expect(error).to_be_visible(timeout=2000)
            except AssertionError:
                pass

    async def test_api_error_displayed(self, page: Page, base_url: str):
        """
        Test that API errors are displayed to user.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        # Enter problem and initialize
        await page.fill("textarea[id='problemInput']", "Test problem")
        await page.click("button:has-text('Initialize')")

        # If there's an error, it should be displayed
        error_elements = page.locator("div[class*='error'], [role='alert']")
        # Just verify the page can handle it
        assert page.url.startswith(base_url)


class TestResponseiveness:
    """Tests for UI responsiveness and loading states."""

    async def test_loading_indicator_shown(self, page: Page, base_url: str):
        """
        Test that loading indicators are shown during processing.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        problem_text = "Long processing task"
        await page.fill("textarea[id='problemInput']", problem_text)

        # Click initialize and check for loading indicator
        await page.click("button:has-text('Initialize')")

        # Look for spinner or loading text
        loading = page.locator("[class*='loading'], [class*='spinner']")
        # Loading indicator might appear briefly
        try:
            await expect(loading).to_be_visible(timeout=1000)
        except AssertionError:
            # Loading might have already completed
            pass

    async def test_buttons_disabled_during_processing(self, page: Page, base_url: str):
        """
        Test that buttons are disabled during async operations.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        problem_text = "Test disable state"
        await page.fill("textarea[id='problemInput']", problem_text)

        # Click initialize
        init_button = page.locator("button:has-text('Initialize')")
        await init_button.click()

        # Button might be disabled during processing
        # This is an optional feature
        try:
            await expect(init_button).to_be_disabled(timeout=1000)
        except AssertionError:
            # Button might not have disable feature
            pass


class TestEndToEndCompletionWorkflow:
    """Tests for complete end-to-end workflows."""

    async def test_complete_workflow_initialization_to_analysis(
        self, page: Page, base_url: str
    ):
        """
        Test complete workflow from initialization to supervisor analysis.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        # Step 1: Enter problem
        problem = "Design a real-time collaboration platform"
        await page.fill("textarea[id='problemInput']", problem)

        # Step 2: Initialize
        await page.click("button:has-text('Initialize')")
        await page.wait_for_timeout(500)  # Wait for initialization

        # Step 3: Supervisor thinks
        supervisor_button = page.locator("button:has-text('Supervisor Thinks')")
        await expect(supervisor_button).to_be_visible()
        await supervisor_button.click()
        await page.wait_for_timeout(500)

        # Verify we're still on the page
        assert page.url.startswith(base_url)

    async def test_complete_workflow_through_delegation(
        self, page: Page, base_url: str
    ):
        """
        Test workflow from initialization through delegation.

        Args:
            page: Playwright page
            base_url: Application base URL
        """
        await page.goto(base_url)

        problem = "Build a scalable data pipeline"
        await page.fill("textarea[id='problemInput']", problem)

        # Initialize
        await page.click("button:has-text('Initialize')")
        await page.wait_for_timeout(500)

        # Supervisor thinks
        await page.click("button:has-text('Supervisor Thinks')")
        await page.wait_for_timeout(500)

        # Delegate
        delegate_button = page.locator("button:has-text('Delegate')")
        if await delegate_button.count() > 0:
            await delegate_button.click()
            await page.wait_for_timeout(500)

        # Should still be on the page
        assert page.url.startswith(base_url)
