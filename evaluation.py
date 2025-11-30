"""
evaluation.py - LLM-as-a-Judge for ADHD Assistant
Evaluates the 'Effectiveness' and 'Robustness' of the agent's planning.
"""
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Import your actual agent architecture
from agents import TaskLogicAgent, ToolExecutionAgent, ConversationManagerAgent

# 1. Setup Environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env!")
genai.configure(api_key=api_key)

# 2. Define the "Golden" Test Case
TEST_CASE = {
    "name": "Decomposition Stress Test",
    "input_text": (
        "I need to apply for a visa by Friday, buy groceries for dinner tonight, "
        "and also email my boss about the project delay."
    ),
    "expected_behavior": "The agent should split this into 3 distinct tasks with different due dates/priorities."
}

def run_evaluation():
    print(f"🧪 STARTING EVALUATION: {TEST_CASE['name']}")
    print("-" * 60)

    # --- A. Run the Agent (The "Subject") ---
    print("🤖 1. Running Agent...")
    
    # CHANGED: Use the working model ID 'gemini-2.5-flash'
    task_agent = TaskLogicAgent(model_name="gemini-2.5-flash") 
    tool_agent = ToolExecutionAgent()
    manager = ConversationManagerAgent(task_agent, tool_agent)

    # Execute the logic (auto_confirm=False to inspect the plan)
    turn = manager.handle_user_message(TEST_CASE["input_text"], user_id="eval_user", auto_confirm=False)
    
    # Extract the "Actual" output to judge
    # We only care about the tasks list for this evaluation
    actual_plan_str = json.dumps([t.__dict__ for t in turn.tasks], indent=2)
    print(f"   -> Agent generated {len(turn.tasks)} tasks.")

    # --- B. Run the Judge (The "Critic") ---
    print("⚖️  2. Running Judge (LLM-as-a-Judge)...")
    
    # Use 'gemini-2.5-flash' for the judge as well to ensure it runs
    judge_model = genai.GenerativeModel("gemini-2.5-flash") 
    
    judge_prompt = f"""
    You are an expert AI Evaluator. Your job is to grade an AI Assistant's performance.
    
    ### THE TASK
    The Assistant receives a messy "brain dump" from a user with ADHD.
    It must decompose this into atomic, clear, and actionable sub-tasks.
    
    ### THE INPUT
    User said: "{TEST_CASE['input_text']}"
    
    ### THE AGENT'S OUTPUT PLAN
    {actual_plan_str}
    
    ### EVALUATION CRITERIA
    1. **Atomicity**: Are the tasks split correctly? (e.g. "Buy groceries" and "Email boss" should be separate).
    2. **Temporal Awareness**: Did it catch the due dates? ("Friday", "Tonight").
    3. **Hallucination**: Did it invent tasks that weren't asked for?
    
    ### YOUR VERDICT
    Provide a JSON response with:
    - "score": An integer from 1-10 (10 is perfect).
    - "reasoning": A brief explanation of why you gave this score.
    - "pass": Boolean (True if score >= 7).
    """
    
    # Get the verdict
    response = judge_model.generate_content(
        judge_prompt,
        generation_config=genai.GenerationConfig(response_mime_type="application/json")
    )
    
    try:
        verdict = json.loads(response.text)
        print("\n" + "="*30)
        print(f"🏆 FINAL SCORE: {verdict['score']}/10")
        print(f"✅ PASSED: {verdict['pass']}")
        print(f"📝 REASONING: {verdict['reasoning']}")
        print("="*30)
    except Exception as e:
        print(f"❌ Error parsing judge response: {e}")
        print(response.text)

if __name__ == "__main__":
    run_evaluation()