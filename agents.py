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
from vertexai.generative_models import GenerationResponse, GenerativeModel, Part, GenerationConfig

from tools import execute_tool

# ---- Vertex AI Initialization -------------------------------------------------------------

try:
    PROJECT_ID = "adhd-assistant-capstone"
    LOCATION = "us-central1"  # Or your desired location
    vertexai.init(project=PROJECT_ID, location=LOCATION)
except Exception as e:
    print(f"ERROR: Vertex AI initialization failed: {e}")
    print(
        "Please make sure you have authenticated with Google Cloud CLI and have the correct permissions."
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

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        # Use 1.5-pro or 2.0-flash for better instruction following
        self.model = GenerativeModel(model_name)

    def decompose_brain_dump(
        self, user_text: str, context: Optional[Dict[str, Any]] = None
    ) -> TaskPlan:
        context = context or {}
        
        # 1. Define the Strict Output Schema
        # This forces the model to fill a LIST of tasks, not just one string
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "tasks": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "description": {"type": "STRING"},
                            "due": {"type": "STRING"},
                            "priority": {"type": "STRING"},
                        },
                        "required": ["description"],
                    },
                },
                "conflicts": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                },
                "encouragement": {"type": "STRING"},
            },
            "required": ["tasks", "encouragement"],
        }

        prompt = self._construct_prompt(user_text, context)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=response_schema # <--- KEY FIX
                ),
            )
            plan = self._parse_model_response(response)
        except Exception as e:
            print(f"Error calling model or parsing response: {e}")
            plan = TaskPlan(
                tasks=[TaskItem(description=user_text)],
                conflicts=["I had trouble decomposing that. Could you list them one by one?"]
            )

        if context.get("encouragement_override"):
            plan.encouragement = context.get("encouragement_override")
            
        return plan

    def _construct_prompt(self, user_text: str, context: Dict[str, Any]) -> str:
        user_preferences = context.get("user_preferences", "No specific preferences.")
        
        return f"""
        You are an expert Task Decomposer for ADHD assistants. 
        Your ONLY goal is to break down complex "brain dumps" into small, atomic, single-action tasks.

        --- RULES FOR DECOMPOSITION ---
        1. **SPLIT COMPOUND SENTENCES**: If a user says "Do X and Y", these MUST be two separate tasks.
        2. **ISOLATE DATES**: If a task has a specific time (e.g., "Friday at 10am"), extract it into the 'due' field.
        3. **BE ATOMIC**: A task description should be short (e.g., "Buy eggs").

        --- ONE-SHOT EXAMPLE ---
        USER: "I need to mail the letter and pick up the dry cleaning tomorrow."
        RESPONSE:
        {{
            "tasks": [
                {{"description": "Mail the letter", "due": null, "priority": "medium"}},
                {{"description": "Pick up dry cleaning", "due": "2025-10-12", "priority": "medium"}}
            ],
            "encouragement": "Two quick errands and you're done!"
        }}
        -------------------------

        --- USER CONTEXT ---
        {user_preferences}
        --------------------

        User's Brain Dump:
        "{user_text}"
        """



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
        user_id: str = "default_user",
        auto_confirm: bool = False,
    ) -> AgentTurn:
        """
        Main entrypoint for user-facing interactions.

        - Retrieves user context via ToolExecutionAgent.
        - Decomposes the brain dump into tasks via TaskLogicAgent.
        - Prepares tool actions (not executed unless auto_confirm=True).
        - Enforces a confirmation loop before scheduling/reminders.
        """
        # 1. Retrieve context (delegating to the "Hands")
        context_action = ToolAction(
            kind="get_user_context",
            payload={"user_id": user_id},
            description="Fetching user context.",
        )
        context_result = self.tool_agent.execute_actions([context_action])
        context = context_result[0].get("context", {})

        # 2. Decompose task (delegating to the "Engine")
        plan = self.task_agent.decompose_brain_dump(user_text=user_text, context=context)

        # 3. Propose actions based on the plan (delegating to the "Hands" again)
        pending_actions = self.tool_agent.propose_actions(plan.tasks)
        requires_confirmation = not auto_confirm

        # 4. Execute actions if confirmation is not required
        if auto_confirm:
            self.tool_agent.execute_actions(pending_actions)

        # 5. Format the final response for the user
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
    agent_turn = manager.handle_user_message(
        user_text=FAKE_USER_BRAIN_DUMP, 
        user_id="user_12345", 
        auto_confirm=True
    )

    print(f"--- AGENT RESPONSE ---\n{agent_turn.user_facing_message}\n")
    print("\n--- TOOL EXECUTION RESULTS ---")
    # In this demo, actions were auto-executed, so we can inspect the (simulated) results.
    # Note that get_user_context was already called inside handle_user_message.
    # The results printed here are for the scheduling and reminder actions.
    for res in agent_turn.pending_actions:
        print(res)

