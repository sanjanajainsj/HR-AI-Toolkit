"""
This script demonstrates how to build an agentic chatbot using LangGraph and Google Gemini. The chatbot is designed to assist HR professionals in assessing the risk of onboarding new candidates by querying a background verification tool. The agent can autonomously decide when to call the tool and how to respond based on the results.

Before running this script, ensure you have the necessary dependencies installed and your GEMINI_API_KEY environment variable set.          
pip install langchain langchain-core langchain-google-genai langgraph

To set your GEMINI_API_KEY environment variable, you can do it in your terminal like this:
export GEMINI_API_KEY="your_actual_api_key_here"
(For Windows users, use: set GEMINI_API_KEY=your_actual_api_key_here)

Feel free to switch this to use OpenAI by replacing the Gemini-specific imports and initialization with OpenAI equivalents, and setting the OPENAI_API_KEY instead.

To run the script, simply execute:
python langgraph_bg_check_chatbot.py

"""


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# Define the Agent's Tools
@tool
def query_background_vendor(candidate_name: str) -> str:
    """Queries the background verification database for a candidate's status."""
    name = candidate_name.lower()
    if "john" in name:
        return "FLAGGED: High risk. Previous employment records show discrepancy."
    return "CLEAR: No risk factors found."


tools = [query_background_vendor]
tool_node = ToolNode(tools)

# Define the State and Initialize LLM
class AgentState(TypedDict):
    # add_messages appends new messages to history instead of overwriting
    messages: Annotated[list, add_messages]

# Initialize the LLM and bind tools to it
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
llm_with_tools = llm.bind_tools(tools)

# Define Node Behaviors
def hr_agent_node(state: AgentState):
    """The central brain. It decides whether to call a tool or finalize the workflow."""
    print("[Agent] Analyzing state and deciding next move...")
    
    system_instruction = SystemMessage(
        content=(
            "You are an AI HR Compliance Agent. Your job is to assess onboarding risk.\n"
            "1. First, check the candidate's background using the `query_background_vendor` tool.\n"
            "2. If the status is CLEAR, instruct the system to 'Generate Standard Employment Offer Letter'.\n"
            "3. If the status is FLAGGED, immediately draft a formal high-risk internal notice email "
            "stating that access is withheld, and state that the application is on hold for human review."
        )
    )
    
    # Prepend system instructions to the conversation history
    messages = [system_instruction] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

# Construct the Graph Topology
workflow = StateGraph(AgentState)

# Define our two operational nodes
workflow.add_node("agent", hr_agent_node)
workflow.add_node("tools", tool_node)

# Map entry point
workflow.add_edge(START, "agent")

# Define dynamic routing: 
# If agent outputs tool_calls -> go to tools. Otherwise -> END.
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)

# After tools run, loop back to the agent so it can read the tool's output
workflow.add_edge("tools", "agent")

# Compile the runnable graph
app = workflow.compile()

# Execution Demos
print("--- RUNNING AGENTIC PATH A: (Jane Smith) ---")
happy_output = app.invoke({
    "messages": [HumanMessage(content="Process onboarding for Jane Smith.")]
})
print(f"\nFinal Agent Response:\n{happy_output['messages'][-1].content}\n")

print("-" * 60)

print("--- RUNNING AGENTIC PATH B: (John Doe) ---")
flagged_output = app.invoke({
    "messages": [HumanMessage(content="Process onboarding for John Doe.")]
})
print(f"\nFinal Agent Response:\n{flagged_output['messages'][-1].content}")