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

# NEW (AI Studio)
import google.generativeai as genai
self.model = genai.GenerativeModel("gemini-1.5-pro")

from tools import execute_tool

# ---- [REMOVED] Vertex AI Initialization ---------------------------------------------------
# We removed the vertexai.init() block here. 
# The initialization is now handled EXCLUSIVELY in the Notebook (Cell 2) 
# using the Service Account. This prevents the credentials from being overwritten.
# -------------------------------------------------------------------------------------------


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
        # gemini-1.5-pro is highly recommended for complex instruction following
        self.model = GenerativeModel(model_name)

    def decompose_brain_dump(
        self, user_text: str, context: Optional[Dict[str, Any]] = None
    ) -> TaskPlan:
        context = context or {}
        
        # 1. Define Strict Output Schema with "Reasoning" field
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "reasoning": {
                    "type": "STRING", 
                    "description": "Step-by-step analysis of the input text to identify distinct actions."
                },
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
            "required": ["reasoning", "tasks", "encouragement"],
        }

        prompt = self._construct_prompt(user_text, context)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.2, 
                    response_mime_type="application/json",
                    response_schema=response_schema
                ),
            )
            plan = self._parse_model_response(response)
        except Exception as e:
            # This print statement is what showed us the Auth error!
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
        Your goal is to turn a chaotic "brain dump" into a clean, atomic checklist.

        --- CRITICAL RULES ---
        1. **ATOMICITY**: Each task must be a SINGLE action. 
           - BAD: "Call doctor and buy bread"
           - GOOD: "Call doctor" (Task 1), "Buy bread" (Task 2)
        2. **NO COPYING**: Do NOT just copy the user's full text into a task. Rewrite it.
        3. **EXTRACT DATES**: If a time is mentioned, move it to the 'due' field.

        --- STRATEGY ---
        1. First, use the "reasoning" field to list out the verbs you see in the text.
        2. Then, create the "tasks" list based on those verbs.

        --- ONE-SHOT EXAMPLE ---
        Input: "I need to mail the letter and pick up the dry cleaning tomorrow."
        Output JSON:
        {{
            "reasoning": "User mentioned two distinct actions: 'mail letter' and 'pick up dry cleaning'. Both have a timeframe of 'tomorrow'.",
            "tasks": [
                {{"description": "Mail the letter", "due": "tomorrow", "priority": "medium"}},
                {{"description": "Pick up dry cleaning", "due": "tomorrow", "priority": "medium"}}
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

    @staticmethod
    def _parse_model_response(response: GenerationResponse) -> TaskPlan:
        try:
            response_dict = response.candidates[0].content.parts[0].json
        except (IndexError, AttributeError, ValueError):
            return TaskPlan(tasks=[], conflicts=["Model response error"])
            
        tasks = [TaskItem(**task_data) for task_data in response_dict.get("tasks", [])]
        return TaskPlan(
            tasks=tasks,
            conflicts=response_dict.get("conflicts", []),
            encouragement=response_dict.get("encouragement", "You got this!"),
        )


class ToolExecutionAgent:
    """The hands: schedules tasks, sets reminders, and retrieves context via MCP tools."""

    def propose_actions(self, tasks: List[TaskItem]) -> List[ToolAction]:
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
            else:
                actions.append(
                    ToolAction(
                        kind="set_reminder",
                        payload={"task_description": task.description, "remind_at": '1 hour from now'},
                        description=f"🔔 Set reminder for '{task.description}'",
                    )
                )
        return actions

    def execute_actions(self, actions: List[ToolAction]) -> List[Any]:
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
        
        # 1. Retrieve context 
        context_action = ToolAction(
            kind="get_user_context",
            payload={"user_id": user_id},
            description="Fetching user context.",
        )
        context_result = self.tool_agent.execute_actions([context_action])
        context = context_result[0].get("context", {})

        # 2. Decompose task
        plan = self.task_agent.decompose_brain_dump(user_text=user_text, context=context)

        # 3. Propose actions
        pending_actions = self.tool_agent.propose_actions(plan.tasks)
        requires_confirmation = not auto_confirm

        # 4. Execute actions if auto_confirm is True
        if auto_confirm:
            self.tool_agent.execute_actions(pending_actions)

        # 5. Format response
        message_parts: List[str] = []
        if plan.encouragement:
            message_parts.append(plan.encouragement)

        if plan.tasks:
            message_parts.append("\nHere's what I've broken down for you:")
            for idx, task in enumerate(plan.tasks, start=1):
                task_details = [f"{idx}. {task.description}"]
                if task.due:
                    task_details.append(f" (Due: {task.due})")
                message_parts.append("".join(task_details))
        
        if pending_actions and requires_confirmation:
            message_parts.append("\nI'll set these up for you:")
            for action in pending_actions:
                message_parts.append(f"- {action.description}")
            message_parts.append("\nSound good?")
        elif not plan.tasks:
            message_parts.append("I couldn't find any specific tasks to list. Could you rephrase?")

        return AgentTurn(
            user_facing_message="\n".join(message_parts),
            tasks=plan.tasks,
            pending_actions=pending_actions,
            requires_confirmation=requires_confirmation,
        )
