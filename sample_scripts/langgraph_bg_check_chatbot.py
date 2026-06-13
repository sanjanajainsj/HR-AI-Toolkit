"""
This script demonstrates how to build a background check chatbot using Langgraph package for Google Gemini. It showcases two main features:
1. Sequential Chains (LCEL): We create a two-step chain where the first step extracts the candidate's target destination from a user query, and the second step generates a localized background check checklist based on that location.
2. Native LLM Tool Calling: We define a custom tool to check a candidate's background status and bind it directly to the Gemini model. The model can autonomously decide when to call the tool based on the user's query.

Before running this script, ensure you have the necessary dependencies installed and your GEMINI_API_KEY environment variable set.

pip install langchain langchain-core langchain-google-genai langgraph

To set your GEMINI_API_KEY environment variable, you can do it in your terminal like this:
export GEMINI_API_KEY="your_actual_api_key_here"

Feel free to switch this to use OpenAI by replacing the Gemini-specific imports and initialization with OpenAI equivalents, and setting the OPENAI_API_KEY instead.

TO run the script, simply execute:
python langgraph_bg_check_chatbot.py
"""


from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

# Define the Central Shared State Structure
class OnboardingState(TypedDict):
    candidate_name: str
    bg_status: Optional[str]
    compliance_action: Optional[str]
    next_step: Optional[str]

# Define the Nodes (The execution blocks)
def query_bg_vendor_node(state: OnboardingState):
    """Node 1: Simulate checking the background status database."""
    name = state["candidate_name"].lower()
    # Mock DB hit
    if "john" in name:
        status = "FLAGGED"
    else:
        status = "CLEAR"
    
    print(f"[Node: Query Vendor] Checked status for {state['candidate_name']}: Result = {status}")
    return {"bg_status": status}

def auto_approve_node(state: OnboardingState):
    """Node 2a: Standard happy path."""
    print("[Node: Auto Approve] Processing standard contract...")
    return {"next_step": "Generate Standard Employment Offer Letter."}

def compliance_escalation_node(state: OnboardingState):
    """Node 2b: Risk/Mitigation path. We invoke an LLM to write an alert email."""
    print("[Node: Compliance Escalation] Warning triggered! Asking LLM to draft high-risk notice...")
    
    prompt = f"Draft a concise internal notification email stating that candidate {state['candidate_name']} has been FLAGGED during background verification and access is withheld."
    llm_response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "compliance_action": llm_response.content,
        "next_step": "Hold application. Awaiting formal human compliance officer review."
    }

# Define Conditional Routing Logic (The Edge Router)
def route_based_on_risk(state: OnboardingState) -> str:
    """This function acts as a conditional edge controller."""
    if state["bg_status"] == "FLAGGED":
        return "trigger_audit"
    return "proceed_to_hire"


# Construct the Graph Topology
workflow = StateGraph(OnboardingState)

# Add our processing nodes
workflow.add_node("check_bg", query_bg_vendor_node)
workflow.add_node("approve_hire", auto_approve_node)
workflow.add_node("escalate_to_compliance", compliance_escalation_node)

# Map the starting point
workflow.set_entry_point("check_bg")

# Add the conditional branching logic from the check_bg node
workflow.add_conditional_edges(
    "check_bg",
    route_based_on_risk,
    {
        "proceed_to_hire": "approve_hire",
        "trigger_audit": "escalate_to_compliance"
    }
)

# Connect terminal endpoints to the explicit END state
workflow.add_edge("approve_hire", END)
workflow.add_edge("escalate_to_compliance", END)

# Compile into a single runnable application
app = workflow.compile()

# Execution Demos
print("--- RUNNING PATH A: THE HAPPY PATH (Jane Smith) ---")
happy_output = app.invoke({"candidate_name": "Jane Smith"})
print(f"Final Outcome: {happy_output['next_step']}\n")

print("-" * 50)

print("--- RUNNING PATH B: THE ESCALATION PATH (John Doe) ---")
flagged_output = app.invoke({"candidate_name": "John Doe"})
print(f"Final Outcome: {flagged_output['next_step']}")
print(f"Drafted Compliance Artifact:\n{flagged_output['compliance_action']}")