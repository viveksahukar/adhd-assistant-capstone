
"""Core agent definitions for the ADHD assistant architecture.

This module defines three collaborating agents:
1. ConversationManagerAgent: orchestrates the flow.
2. TaskLogicAgent: decomposes user intent (Using Google AI Studio / Free Tier).
3. ToolExecutionAgent: prepares and executes tool calls.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# --- CHANGED: Import Google AI Studio library instead of Vertex AI ---
import google.generativeai as genai
from google.ai.generativelanguage import Content, Part

from tools import execute_tool

# ---- Shared data models --------------------------------------------------------------------

@dataclass
class TaskItem:
    description: str
    status: str = "pending"
    due: Optional[str] = None
    priority: Optional[str] = None
    conflicts: List[str] = field(default_factory=list)

@dataclass
class TaskPlan:
    tasks: List[TaskItem]
    encouragement: Optional[str] = None
    conflicts: List[str] = field(default_factory=list)

@dataclass
class ToolAction:
    kind: str
    payload: Dict[str, Any]
    description: str

@dataclass
class AgentTurn:
    user_facing_message: str
    tasks: List[TaskItem] = field(default_factory=list)
    pending_actions: List[ToolAction] = field(default_factory=list)
    requires_confirmation: bool = True

# ---- Agents --------------------------------------------------------------------------------

class TaskLogicAgent:
    """The engine: decomposes user intent using the Free Tier API."""

    def __init__(self, model_name: str = "gemini-1.5-pro"):
        # CHANGED: Use genai.GenerativeModel (Free Tier)
        self.model = genai.GenerativeModel(model_name)

    def decompose_brain_dump(
        self, user_text: str, context: Optional[Dict[str, Any]] = None
    ) -> TaskPlan:
        context = context or {}
        
        prompt = self._construct_prompt(user_text, context)
        
        try:
            # CHANGED: API call syntax for Google AI Studio
            # We request JSON response_mime_type for structured output
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            plan = self._parse_model_response(response)
        except Exception as e:
            print(f"Error calling model or parsing response: {e}")
            # Fallback
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
        GOAL: Break down the user's text into atomic tasks.

        OUTPUT SCHEMA (JSON):
        {{
            "reasoning": "Step-by-step analysis string",
            "tasks": [
                {{
                    "description": "Short task description", 
                    "due": "Due date or null", 
                    "priority": "high/medium/low"
                }}
            ],
            "conflicts": ["List of potential conflicts strings"],
            "encouragement": "Encouraging message string"
        }}

        --- USER CONTEXT ---
        {user_preferences}
        --------------------

        User's Brain Dump:
        "{user_text}"
        """

    @staticmethod
    def _parse_model_response(response) -> TaskPlan:
        try:
            # CHANGED: Parsing logic for Google AI Studio response object
            response_text = response.text
            response_dict = json.loads(response_text)
        except Exception:
            return TaskPlan(tasks=[], conflicts=["Model response error"])
            
        tasks = [TaskItem(**task_data) for task_data in response_dict.get("tasks", [])]
        return TaskPlan(
            tasks=tasks,
            conflicts=response_dict.get("conflicts", []),
            encouragement=response_dict.get("encouragement", "You got this!"),
        )


class ToolExecutionAgent:
    """The hands: schedules tasks, sets reminders."""

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
    """The face/orchestrator."""

    def __init__(self, task_agent: TaskLogicAgent, tool_agent: ToolExecutionAgent) -> None:
        self.task_agent = task_agent
        self.tool_agent = tool_agent

    def handle_user_message(
        self,
        user_text: str,
        user_id: str = "default_user",
        auto_confirm: bool = False,
    ) -> AgentTurn:
        
        context_action = ToolAction(
            kind="get_user_context",
            payload={"user_id": user_id},
            description="Fetching user context.",
        )
        context_result = self.tool_agent.execute_actions([context_action])
        context = context_result[0].get("context", {})

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
