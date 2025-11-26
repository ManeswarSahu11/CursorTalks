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
    Takes a command line prompt and executes on the user's machine and returns the 
    output of the command.
    Example: 
    run_command(command="ls") where ls is the command to list the files.
    """
    try:
        result = os.popen(command).read()
        return f"Command executed successfully. Output:\n{result}" if result else "Command executed successfully (no output)."
    except Exception as e:
        return f"Error executing command: {str(e)}"

llm = init_chat_model("openai:gpt-4o")
tools = [run_command]
llm_with_tools = llm.bind_tools(tools=tools)

def enhance_query(state: State):
    """Understands and break down the user's query for better understanding"""
    original_query = state["messages"][-1].content if state["messages"] else ""
    
    enhancement_prompt = SystemMessage(
        content="""
        You are an AI assistant specialised in understanding queries of user and breaking it down into a proper plan for programming. 
        Your job is to take a user's question and rewrite it in a clear, specific, and actionable manner.
        
        Guidelines:
        - If the query is about programming, coding, development, debugging, or software engineering, improve it.
        - Make the query more specific and actionable.
        - Include context about what the user likely wants to achieve.
        - If the query is NOT about programming (like general chat, personal questions, etc.), 
          respond with "NON_PROGRAMMING_QUERY"
        
        Examples:
        Input: "write a python code to add 2 numbers"
        Output: "Create a Python script that prompts the user to input two numbers, performs addition and subtraction operations, and displays the results with clear output messages."
        
        Input: "what's the weather today in Delhi?"
        Output: "NON_PROGRAMMING_QUERY"        
        """
    )
    
    response = llm.invoke([enhancement_prompt, HumanMessage(content=f"Enhance this query: {original_query}")])
    enhanced_query = response.content.strip()
    
    return {"enhanced_query": enhanced_query}

def create_plan(state: State):
    """Creates a step-by-step plan for solving the user's task"""
    enhanced_query = state.get("enhanced_query", "")
    
    if enhanced_query == "NON_PROGRAMMING_QUERY":
        return {
            "plan": ["REJECT_NON_PROGRAMMING"],
            "current_step": 0
        }
    
    planning_prompt = SystemMessage(
        content="""
        You are a expert AI assistant who creates executable action plans for programming tasks.
        
        Create a plan with SPECIFIC EXECUTABLE STEPS that involve actual file creation.
        
        Guidelines:
        - Step 1: Always "Create ai_solution directory" 
        - Step 2: Always "Create [specific_filename].py file with [specific functionality]"
        - Step 3: Always "Test the created Python file by running it"
        - Each step must be actionable with specific filenames and functionality
        - MAXIMUM 3 steps focused on: directory creation, file creation, testing
        
        Example plan format:
        1. Create ai_solution directory using mkdir command
        2. Create add_numbers.py file with input prompts and calculation logic 
        3. Test the add_numbers.py file by executing it
        
        Format your response as a numbered list, one step per line.
        """
    )
    
    response = llm.invoke([planning_prompt, HumanMessage(content=f"Create an executable plan for: {enhanced_query}")])
    plan_text = response.content.strip()
    
    # Parse the plan into a list
    plan_steps = []
    for line in plan_text.split('\n'):
        if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
            # Remove numbering and clean up
            step = re.sub(r'^\d+\.?\s*', '', line.strip())
            step = re.sub(r'^-\s*', '', step.strip())
            if step:
                plan_steps.append(step)
    
    return {
        "plan": plan_steps,
        "current_step": 0,
        "execution_summary": ""
    }

def execute_step(state: State):
    """Executes the current step of the plan"""
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    
    if not plan:
        return {"messages": [AIMessage(content="No plan available to execute.")]}
    
    if plan[0] == "REJECT_NON_PROGRAMMING":
        return {
            "messages": [AIMessage(content="I can only answer programming-related queries. Please ask me about coding, software development, debugging, or other technical programming topics.")],
            "execution_summary": "Non-programming query rejected",
            "current_step": len(plan)  # Mark as completed to trigger summary
        }
    
    if current_step >= len(plan):
        # All steps completed, generate summary
        summary = f"Completed all planned steps. Executed {len(plan)} steps to address the programming task."
        return {
            "execution_summary": summary,
            "current_step": current_step
        }
    
    current_step_description = plan[current_step]
    
    system_prompt = SystemMessage(
        content=f"""
        You are a world-class AI coding assistant. Your job is to EXECUTE commands, not explain them.
        
        MANDATORY EXECUTION RULES:
        - You MUST call the run_command tool for EVERY action
        - You MUST create actual files and directories
        - DO NOT provide explanations without executing commands first
        - ALWAYS execute the step immediately using run_command
        
        CURRENT TASK: "{current_step_description}"
        
        REQUIRED ACTIONS FOR THIS STEP:
        1. If creating directory: IMMEDIATELY call run_command with "mkdir -p ai_solution"
        2. If creating Python file: IMMEDIATELY call run_command with appropriate file creation command
        3. If writing code: IMMEDIATELY call run_command to write the actual code to file
        
        FILE CREATION COMMANDS YOU MUST USE:
        - For directories: run_command("mkdir -p ai_solution")
        - For Python files: run_command('cat > ai_solution/filename.py << "EOF"\n[PYTHON CODE HERE]\nEOF')
        - For testing: run_command("python ai_solution/filename.py")
        
        CRITICAL: You must EXECUTE commands now, not describe what to do. Start by calling run_command immediately.
        """
    )
    
    # Get the original user message for context
    user_message = ""
    for msg in reversed(state["messages"]):
        if hasattr(msg, 'content') and msg.content:
            user_message = msg.content
            break
    
    context_message = HumanMessage(
        content=f"""EXECUTE THIS IMMEDIATELY: {current_step_description}

Original user request: {user_message}

YOU MUST CALL run_command RIGHT NOW. Do not explain, do not describe - EXECUTE the command immediately.

If this step involves:
- Creating directory → Call run_command("mkdir -p ai_solution") 
- Creating Python file → Call run_command with cat command to write the file
- Writing code → Call run_command to actually write code to file
- Testing → Call run_command to run the Python file

EXECUTE NOW!"""
    )
    
    response = llm_with_tools.invoke([system_prompt, context_message])
    
    return {
        "messages": [response],
        "current_step": current_step + 1
    }

def should_continue_execution(state: State) -> Literal["continue", "tools", "complete"]:
    """Determine if we should continue execution, use tools, or complete"""
    if state.get("awaiting_confirmation", False):
        return "complete"
    
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])
    
    if not plan or current_step >= len(plan):
        return "complete"
    
    # Check if the last message has tool calls
    if state.get("messages") and hasattr(state["messages"][-1], 'tool_calls') and state["messages"][-1].tool_calls:
        return "tools"
    
    return "continue"

def generate_summary_and_speak(state: State):
    """Generates summary of actions taken"""
    execution_summary = state.get("execution_summary", "")
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    
    # Check if this was a rejected non-programming query
    if plan and plan[0] == "REJECT_NON_PROGRAMMING":
        summary = "I can only answer programming-related queries. Please ask me about coding, software development, debugging, or other technical programming topics."
        spoken_summary = f"{summary} What programming question can I help you with?"
        
        return {
            "execution_summary": summary,
            "messages": [AIMessage(content=spoken_summary)]
        }
    
    # Handle regular programming tasks
    if not execution_summary and plan:
        completed_steps = min(current_step, len(plan))
        # Create a concise summary (50-200 words)
        summary = f"Task completed successfully! I executed {completed_steps} steps: "
        summary += "; ".join(plan[:completed_steps])
        summary += ". All files have been created in the ai_solution folder and are ready to use."
        
        # Keep it within 50-200 words
        if len(summary.split()) > 200:
            summary = f"Task completed! I successfully executed all {completed_steps} planned steps for your programming request. The solution has been implemented and files are ready in the ai_solution folder."
    else:  
        summary = execution_summary or "Programming task processing completed."
    
    # Also provide information about what was created
    summary_with_files = f"{summary} You can find your Python script in the ai_solution directory and run it to see the results."
        
    return {
        "execution_summary": summary_with_files,
        "messages": [AIMessage(content=summary_with_files)]
    }

# Create the tool node
tool_node = ToolNode(tools=tools)

# Build the graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("enhance_query", enhance_query)
graph_builder.add_node("create_plan", create_plan)
graph_builder.add_node("execute_step", execute_step)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("generate_summary", generate_summary_and_speak)

# Add edges
graph_builder.add_edge(START, "enhance_query")
graph_builder.add_edge("enhance_query", "create_plan")
graph_builder.add_edge("create_plan", "execute_step")

# Add conditional edges for execution flow
graph_builder.add_conditional_edges(
    "execute_step",
    should_continue_execution,
    {
        "continue": "execute_step",  # Loop back to execute next step
        "tools": "tools",            # Go to tools if tool calls are needed
        "complete": "generate_summary"
    }
)

graph_builder.add_edge("tools", "execute_step")  # After tools, continue execution
graph_builder.add_edge("generate_summary", END)

def create_chat_graph(checkpointer):
    """Creates a new graph with checkpointer"""
    return graph_builder.compile(checkpointer=checkpointer)