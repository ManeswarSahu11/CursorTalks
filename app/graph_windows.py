from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
import os
import re
from dotenv import load_dotenv  
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import asyncio
import subprocess

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    enhanced_query: str
    plan: list
    current_step: int
    execution_summary: str
    awaiting_confirmation: bool
    dangerous_command: str

@tool
def run_command(command: str):
    """
    Executes a shell command or handles file creation via special syntax.
    Use 'create_file("filepath", "file contents")' for cross-platform file creation.
    """
    try:
        if command.startswith("create_file("):
            match = re.match(r'create_file\(["\'](.+?)["\'],\s*["\']([\s\S]*?)["\']\)', command)
            if not match:
                return "Invalid create_file format."

            filepath, content = match.groups()
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"✅ File '{filepath}' created successfully."

        result = subprocess.run(command, capture_output=True, shell=True, text=True)
        return result.stdout.strip() or result.stderr.strip() or "✅ Command executed (no output)."

    except Exception as e:
        return f"❌ Error executing command: {str(e)}"

llm = init_chat_model("openai:gpt-4o")
tools = [run_command]
llm_with_tools = llm.bind_tools(tools=tools)

def enhance_query(state: State):
    original_query = state["messages"][-1].content if state["messages"] else ""
    enhancement_prompt = SystemMessage(
      content="""
        You are a helpful coding assistant.
        Rewrite the user's request into a specific software engineering task that includes:
        - The goal of the task
        - The programming language
        - The name of the file to be created
        - The function name (if applicable)
        - The input and output expectations

        Return a single enhanced query that can be executed by an automated assistant.

        If the task is NOT about programming or software, return exactly: NON_PROGRAMMING_QUERY
      """
    )
    response = llm.invoke([enhancement_prompt, HumanMessage(content=f"Enhance this query: {original_query}")])
    return {"enhanced_query": response.content.strip()}

def create_plan(state: State):
    enhanced_query = state.get("enhanced_query", "")
    if enhanced_query == "NON_PROGRAMMING_QUERY":
        return {"plan": ["REJECT_NON_PROGRAMMING"], "current_step": 0}

    planning_prompt = SystemMessage(
    content="""
      You are an expert software assistant.

      Create a clear 3-step plan to implement the user's programming task.

      Keep the code modular in structure while creating Python files.

      Each step must be directly executable. Use format:
      1. Create ai_solution directory
      2. Create <filename>.py with <what it should contain> (e.g., function definition, input/output handling, etc.)
      3. Run <filename>.py to verify functionality

      Return only the steps in this format.
      """
      )

    response = llm.invoke([planning_prompt, HumanMessage(content=f"Create a plan for: {enhanced_query}")])
    plan_text = response.content.strip()

    plan_steps = []
    for line in plan_text.split('\n'):
        if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
            step = re.sub(r'^\d+\.?\s*', '', line.strip())
            step = re.sub(r'^-\s*', '', step.strip())
            if step:
                plan_steps.append(step)

    return {"plan": plan_steps, "current_step": 0, "execution_summary": ""}

def execute_step(state: State):
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)

    if not plan:
        return {"messages": [AIMessage(content="No plan available to execute.")]}

    if plan[0] == "REJECT_NON_PROGRAMMING":
        return {
            "messages": [AIMessage(content="I can only answer programming-related queries.")],
            "execution_summary": "Non-programming query rejected",
            "current_step": len(plan)
        }

    if current_step >= len(plan):
        summary = f"Completed all planned steps. Executed {len(plan)} steps."
        return {"execution_summary": summary, "current_step": current_step}

    current_step_description = plan[current_step]
    user_message = next((msg.content for msg in reversed(state["messages"]) if hasattr(msg, 'content')), "")

    system_prompt = SystemMessage(
    content=f"""
      You are a coding assistant executing programming steps.

      Important:
      - Use run_command("create_file('ai_solution/filename.py', 'code')") to create files.
      - The file must contain complete, functional code with all necessary imports and a __main__ section if needed.
      - If the file already exists, overwrite it.

      Current Step: "{current_step_description}"
      """
    )
    
    context_message = HumanMessage(
        content=f"EXECUTE: {current_step_description}\nOriginal user request: {user_message}"
    )
    response = llm_with_tools.invoke([system_prompt, context_message])
    return {"messages": [response], "current_step": current_step + 1}

def should_continue_execution(state: State) -> Literal["continue", "tools", "complete"]:
    if state.get("awaiting_confirmation", False):
        return "complete"
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])
    if not plan or current_step >= len(plan):
        return "complete"
    if state.get("messages") and hasattr(state["messages"][-1], 'tool_calls') and state["messages"][-1].tool_calls:
        return "tools"
    return "continue"

def generate_summary_and_speak(state: State):
    execution_summary = state.get("execution_summary", "")
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    if plan and plan[0] == "REJECT_NON_PROGRAMMING":
        summary = "Only programming queries are allowed."
    else:
        completed_steps = min(current_step, len(plan))
        summary = f"Executed {completed_steps} steps: {', '.join(plan[:completed_steps])}. Files are in ai_solution folder."
    return {"execution_summary": summary, "messages": [AIMessage(content=summary)]}

tool_node = ToolNode(tools=tools)
graph_builder = StateGraph(State)
graph_builder.add_node("enhance_query", enhance_query)
graph_builder.add_node("create_plan", create_plan)
graph_builder.add_node("execute_step", execute_step)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("generate_summary", generate_summary_and_speak)
graph_builder.add_edge(START, "enhance_query")
graph_builder.add_edge("enhance_query", "create_plan")
graph_builder.add_edge("create_plan", "execute_step")
graph_builder.add_conditional_edges("execute_step", should_continue_execution, {
    "continue": "execute_step",
    "tools": "tools",
    "complete": "generate_summary"
})
graph_builder.add_edge("tools", "execute_step")
graph_builder.add_edge("generate_summary", END)

def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
