"""Core agent definitions for the ADHD assistant architecture.

This module defines three collaborating agents:
1. ConversationManagerAgent: orchestrates the flow and enforces HITL confirmation.
2. TaskLogicAgent: decomposes user intent into atomic tasks and checks for conflicts.
3. ToolExecutionAgent: prepares and executes tool calls (calendar, reminders, memory).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import vertexai
from vertexai.generative_models import GenerationResponse, GenerativeModel, Part

from tools import execute_tool

# ---- Vertex AI Initialization -------------------------------------------------------------

try:
    PROJECT_ID = os.environ["adhd-assistant-capstone"]
    LOCATION = "us-central1"  # Or your desired location
    vertexai.init(project=PROJECT_ID, location=LOCATION)
except KeyError:
    print(
        "WARNING: GOOGLE_CLOUD_PROJECT environment variable not set. "
        "Vertex AI features will not be available."
    )

# ---- Shared data models --------------------------------------------------------------------


@dataclass
class TaskItem:
    """Lightweight representation of an atomic task."""

    description: str
    status: str = "pending"
    due: Optional[str] = None
    priority: Optional[str] = None
    conflicts: List[str] = field(default_factory=list)


@dataclass
class TaskPlan:
    """Structured output from the TaskLogicAgent."""

    tasks: List[TaskItem]
    encouragement: Optional[str] = None
    conflicts: List[str] = field(default_factory=list)


@dataclass
class ToolAction:
    """Deferred tool action that requires user confirmation (HITL)."""

    kind: str
    payload: Dict[str, Any]
    description: str


@dataclass
class AgentTurn:
    """Response envelope returned by the ConversationManagerAgent."""

    user_facing_message: str
    tasks: List[TaskItem] = field(default_factory=list)
    pending_actions: List[ToolAction] = field(default_factory=list)
    requires_confirmation: bool = True


# ---- Agents --------------------------------------------------------------------------------


class TaskLogicAgent:
    """The engine: decomposes user intent into atomic tasks and checks conflicts."""

    def __init__(self, model_name: str = "gemini-1.0-pro-001"):
        self.model = GenerativeModel(model_name)

    def decompose_brain_dump(
        self, user_text: str, context: Optional[Dict[str, Any]] = None
    ) -> TaskPlan:
        """
        Uses a generative model to decompose a user's "brain dump" into structured tasks.
        """
        context = context or {}
        prompt = self._construct_prompt(user_text, context)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
            )
            plan = self._parse_model_response(response)
        except Exception as e:
            print(f"Error calling model or parsing response: {e}")
            # Fallback to a simple plan on error
            plan = TaskPlan(
                tasks=[TaskItem(description=user_text)],
                conflicts=["I had trouble understanding that. Could you rephrase?"]
            )

        # Allow context to override the model's generated encouragement
        if context.get("encouragement_override"):
            plan.encouragement = context.get("encouragement_override")
            
        return plan

    def _construct_prompt(self, user_text: str, context: Dict[str, Any]) -> str:
        # We can inject more context here later (e.g., existing tasks, user preferences)
        return f"""
        You are an expert at helping users with ADHD break down a "brain dump" of text into a clear, actionable task list.
        Analyze the user's text and extract distinct tasks.

        Respond in this exact JSON format:
        {{
          "tasks": [{{ "description": "A short, clear description of the task.", "due": "An optional due date if mentioned, in ISO 8601 format.", "priority": "An optional priority (low, medium, high)." }}],
          "conflicts": ["A list of any potential conflicts or ambiguities you found, like duplicate tasks."],
          "encouragement": "A brief, positive, and encouraging message for the user."
        }}

        User's brain dump:
        ---
        {user_text}
        ---
        """

    @staticmethod
    def _parse_model_response(response: GenerationResponse) -> TaskPlan:
        response_dict = response.candidates[0].content.parts[0].json
        tasks = [TaskItem(**task_data) for task_data in response_dict.get("tasks", [])]
        return TaskPlan(
            tasks=tasks,
            conflicts=response_dict.get("conflicts", []),
            encouragement=response_dict.get("encouragement", "You got this! Let's get these organized."),
        )


class ToolExecutionAgent:
    """The hands: schedules tasks, sets reminders, and retrieves context via MCP tools."""

    def propose_actions(self, tasks: List[TaskItem]) -> List[ToolAction]:
        """Prepare tool actions without executing them; keeps HITL in the loop."""
        actions: List[ToolAction] = []
        for task in tasks:
            if task.due:
                actions.append(
                    ToolAction(
                        kind="schedule_event",
                        payload={"task_description": task.description, "due_date": task.due, "priority": task.priority or 'normal'},
                        description=f"✅ Schedule '{task.description}' for {task.due}",
                    )
                )
            actions.append(
                ToolAction(
                    kind="set_reminder",
                    payload={"task_description": task.description, "remind_at": task.due or '1 hour from now'},
                    description=f"🔔 Set reminder for '{task.description}'",
                )
            )
        return actions

    def execute_actions(self, actions: List[ToolAction]) -> List[Any]:
        """Execute prepared actions. Replace _noop with real MCP tool integrations."""
        results: List[Any] = []
        for action in actions:
            try:
                result = execute_tool(action.kind, action.payload)
                results.append(result)
            except Exception as e:
                print(f"Error executing action '{action.kind}': {e}")
                results.append({"status": "error", "details": str(e)})
        return results



class ConversationManagerAgent:
    """The face/orchestrator: manages the session and overall flow."""

    def __init__(self, task_agent: TaskLogicAgent, tool_agent: ToolExecutionAgent) -> None:
        self.task_agent = task_agent
        self.tool_agent = tool_agent

    def handle_user_message(
        self,
        user_text: str,
        context: Optional[Dict[str, Any]] = None,
        auto_confirm: bool = False,
    ) -> AgentTurn:
        """
        Main entrypoint for user-facing interactions.

        - Decomposes the brain dump into tasks via TaskLogicAgent.
        - Prepares tool actions (not executed unless auto_confirm=True).
        - Enforces a confirmation loop before scheduling/reminders.
        """
        plan = self.task_agent.decompose_brain_dump(user_text=user_text, context=context)
        pending_actions = self.tool_agent.propose_actions(plan.tasks)
        requires_confirmation = not auto_confirm

        if auto_confirm:
            self.tool_agent.execute_actions(pending_actions)

        message_parts: List[str] = []
        if plan.encouragement:
            message_parts.append(plan.encouragement)

        if plan.tasks:
            message_parts.append("\nHere's what I've broken down for you:")
            for idx, task in enumerate(plan.tasks, start=1):
                task_details = [f"{idx}. {task.description}"]
                if task.due:
                    task_details.append(f" (Due: {task.due})")
                if task.priority:
                    task_details.append(f" [Priority: {task.priority}]")
                message_parts.append("".join(task_details))
        
        if plan.conflicts:
            message_parts.append("\nI noticed a few things to double-check:")
            for conflict in plan.conflicts:
                message_parts.append(f"• {conflict}")

        if pending_actions and requires_confirmation:
            message_parts.append("\nI can help with the following:")
            for action in pending_actions:
                message_parts.append(f"- {action.description}")
            message_parts.append("\nShall I proceed?")
        elif not plan.tasks:
            message_parts.append("I didn't find any specific tasks. Can you give me a bit more detail?")


        return AgentTurn(
            user_facing_message="\n".join(message_parts),
            tasks=plan.tasks,
            pending_actions=pending_actions,
            requires_confirmation=requires_confirmation,
        )


# ---- Example Usage -------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Initialize the agents
    task_agent = TaskLogicAgent()
    tool_agent = ToolExecutionAgent()
    manager = ConversationManagerAgent(task_agent=task_agent, tool_agent=tool_agent)

    # 2. Simulate a user "brain dump"
    # FAKE_USER_BRAIN_DUMP = (
    #     "I need to finish the report for work by Friday, schedule a dentist appointment for next week,"
    #     " and don't forget to buy milk. Also, I should really start meditating."
    # )
    #
    # # 3. Handle the user message (with confirmation loop)
    # print(f"--- USER INPUT ---\n{FAKE_USER_BRAIN_DUMP}\n")
    # agent_turn = manager.handle_user_message(FAKE_USER_BRAIN_DUMP)
    #
    # print(f"--- AGENT RESPONSE ---\n{agent_turn.user_facing_message}\n")
    #
    # # 4. Simulate user confirmation and execute actions
    # if agent_turn.requires_confirmation:
    #     # In a real app, you would wait for user input here.
    #     print("--- USER CONFIRMS ---\n")
    #     results = tool_agent.execute_actions(agent_turn.pending_actions)
    #     print("\n--- TOOL EXECUTION RESULTS ---")
    #     for res in results:
    #         print(res)
    
    # 2. Simulate a user "brain dump"
    FAKE_USER_BRAIN_DUMP = (
        "I need to finish the report for work by Friday, schedule a dentist appointment for next week,"
        " and don't forget to buy milk. Also, I should really start meditating."
    )

    # 3. Handle the user message (with auto-confirm for demonstration)
    print(f"--- USER INPUT ---\n{FAKE_USER_BRAIN_DUMP}\n")
    agent_turn = manager.handle_user_message(FAKE_USER_BRAIN_DUMP, auto_confirm=True)

    print(f"--- AGENT RESPONSE ---\n{agent_turn.user_facing_message}\n")
    print("\n--- TOOL EXECUTION RESULTS ---")
    # In this demo, actions were auto-executed, so we can inspect the (simulated) results.
    # The `execute_actions` method in a real scenario would be called after user confirmation.
    
    # To show the results, we will re-execute the actions.
    # In a real application, the results would be available from the initial `handle_user_message` call if `auto_confirm` is True.
    results = tool_agent.execute_actions(agent_turn.pending_actions)
    for res in results:
        print(res)

