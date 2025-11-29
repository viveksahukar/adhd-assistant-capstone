"""Module for executing tools and external actions.

This module provides a centralized function `execute_tool` that acts as a dispatcher
for various external tools, such as calendar management, reminder systems, or any
other external API integration.

The primary function is:
- execute_tool: A dispatcher that takes a tool name and a payload and routes
  the call to the appropriate tool implementation.

Each tool function is expected to handle its own specific logic, including API calls,
error handling, and data transformation. The `execute_tool` function provides a
consistent interface for the agents to interact with these tools.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict

# ---- Tool Implementations ------------------------------------------------------------------

def schedule_event(
    task_description: str, due_date: str, priority: str = "normal"
) -> Dict[str, Any]:
    """
    Schedules an event in a calendar.

    In a real implementation, this would interact with a service like Google Calendar.

    Args:
        task_description: The description of the event to schedule.
        due_date: The due date of the event in ISO 8601 format.
        priority: The priority of the event (e.g., 'high', 'normal', 'low').

    Returns:
        A dictionary containing the result of the scheduling operation.
    """
    print(f"--- TOOL: Scheduling event: '{task_description}' for {due_date} with {priority} priority ---")
    # Simulate API call
    return {
        "status": "success",
        "event_id": f"evt_{datetime.datetime.now().isoformat()}",
        "details": f"Event '{task_description}' scheduled for {due_date}.",
    }

def set_reminder(task_description: str, remind_at: str) -> Dict[str, Any]:
    """
    Sets a reminder for a task.

    In a real implementation, this would interact with a service like Google Keep or a mobile notification system.

    Args:
        task_description: The description of the reminder.
        remind_at: The time to send the reminder in ISO 8601 format.

    Returns:
        A dictionary containing the result of the reminder operation.
    """
    print(f"--- TOOL: Setting reminder: '{task_description}' at {remind_at} ---")
    # Simulate API call
    return {
        "status": "success",
        "reminder_id": f"rem_{datetime.datetime.now().isoformat()}",
        "details": f"Reminder for '{task_description}' set for {remind_at}.",
    }

# ---- Tool Dispatcher -----------------------------------------------------------------------

# Mapping of tool names to their implementation functions.
TOOL_REGISTRY = {
    "schedule_event": schedule_event,
    "set_reminder": set_reminder,
}

def execute_tool(tool_name: str, payload: Dict[str, Any]) -> Any:
    """
    Executes a specified tool with a given payload.

    This function acts as a dispatcher, looking up the tool in the TOOL_REGISTRY
    and executing it with the provided arguments.

    Args:
        tool_name: The name of the tool to execute (e.g., 'schedule_event').
        payload: A dictionary of arguments to pass to the tool function.

    Returns:
        The result of the tool's execution.

    Raises:
        ValueError: If the specified tool is not found in the registry.
    """

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool_function = TOOL_REGISTRY[tool_name]
    return tool_function(**payload)
