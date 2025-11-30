"""
tools.py - Persistent Tools for ADHD Assistant
Now with real file I/O to simulate database/API persistence.
"""
import json
import os
import datetime
from typing import Any, Dict, List

# --- Configuration ---
USER_PROFILE_FILE = "user_profile.json"
CALENDAR_DB_FILE = "calendar_db.json"

# --- Helper Functions ---
def _load_json(filepath: str) -> Any:
    """Helper to safely load JSON data."""
    if not os.path.exists(filepath):
        return {} if filepath == USER_PROFILE_FILE else []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {} if filepath == USER_PROFILE_FILE else []

def _save_json(filepath: str, data: Any):
    """Helper to safely save JSON data."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

# --- Tool Implementations ---

def get_user_context(user_id: str) -> Dict[str, Any]:
    """
    Retrieves user profile and preferences from local storage.
    Simulates a RAG retrieval or Database lookup.
    """
    print(f"🧠 [MEMORY] Reading profile for: {user_id}")
    
    data = _load_json(USER_PROFILE_FILE)
    
    # In a real app, we would filter by user_id. 
    # For this prototype, we assume the file belongs to the single active user.
    
    # Format it as a string for the LLM to read easily
    prefs = data.get("preferences", {})
    context_str = (
        f"User Name: {data.get('name', 'Unknown')}. "
        f"Focus Time: {prefs.get('focus_time', 'Unknown')}. "
        f"Style: {prefs.get('communication_style', 'Standard')}. "
        f"Current Goals: {', '.join(data.get('goals', []))}."
    )
    
    return {
        "status": "success",
        "context": {
            "user_preferences": context_str,
            "raw_profile": data # useful for debugging
        }
    }

def schedule_event(task_description: str, due_date: str, priority: str = "normal") -> Dict[str, Any]:
    """
    Adds an event to the persistent calendar file.
    """
    print(f"📅 [CALENDAR] Scheduling '{task_description}'...")
    
    events = _load_json(CALENDAR_DB_FILE)
    
    new_event = {
        "id": f"evt_{len(events) + 1}",
        "title": task_description,
        "due": due_date,
        "priority": priority,
        "status": "scheduled",
        "created_at": datetime.datetime.now().isoformat()
    }
    
    events.append(new_event)
    _save_json(CALENDAR_DB_FILE, events)
    
    return {
        "status": "success",
        "event_id": new_event["id"],
        "details": f"Scheduled '{task_description}' for {due_date}. Total events: {len(events)}"
    }

def set_reminder(task_description: str, remind_at: str) -> Dict[str, Any]:
    """
    Logs a reminder. (For simplicity, we save these to the calendar DB too).
    """
    print(f"⏰ [REMINDER] Setting reminder for '{task_description}'...")
    
    events = _load_json(CALENDAR_DB_FILE)
    
    new_reminder = {
        "id": f"rem_{len(events) + 1}",
        "title": f"REMINDER: {task_description}",
        "due": remind_at,
        "type": "notification",
        "created_at": datetime.datetime.now().isoformat()
    }
    
    events.append(new_reminder)
    _save_json(CALENDAR_DB_FILE, events)

    return {
        "status": "success",
        "reminder_id": new_reminder["id"],
        "details": f"Reminder set for '{task_description}' at {remind_at}"
    }

# --- Dispatcher ---
TOOL_REGISTRY = {
    "schedule_event": schedule_event,
    "set_reminder": set_reminder,
    "get_user_context": get_user_context,
}

def execute_tool(tool_name: str, payload: Dict[str, Any]) -> Any:
    if tool_name not in TOOL_REGISTRY:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    return TOOL_REGISTRY[tool_name](**payload)