"""
This script demonstrates how to build a background check chatbot using LangChain's partner package for Google Gemini. It showcases two main features:
1. Sequential Chains (LCEL): We create a two-step chain where the first step extracts the candidate's target destination from a user query, and the second step generates a localized background check checklist based on that location.
2. Native LLM Tool Calling: We define a custom tool to check a candidate's background status and bind it directly to the Gemini model. The model can autonomously decide when to call the tool based on the user's query.

Before running this script, ensure you have the necessary dependencies installed and your GEMINI_API_KEY environment variable set.

pip install langchain langchain-core langchain-google-genai langgraph

To set your GEMINI_API_KEY environment variable, you can do it in your terminal like this:
export GEMINI_API_KEY="your_actual_api_key_here"

Feel free to switch this to use OpenAI by replacing the Gemini-specific imports and initialization with OpenAI equivalents, and setting the OPENAI_API_KEY instead.

To run the script, simply execute:
python langchain_bg_check_chatbot.py

"""


import os
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the Gemini model via LangChain's partner package
# (Ensure GEMINI_API_KEY environment variable is set)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

# =====================================================================
# 1. Sequential Chain (LCEL): Extracting Profile -> Fetching Checklist
# =====================================================================
print("--- SEQUENTIAL CHAINS (LCEL) ---")

# Chain Step A: Extract destination data
extract_prompt = ChatPromptTemplate.from_template(
    "Extract the candidate's target destination country/state from this query: {user_query}. Return ONLY the location name."
)
extract_chain = extract_prompt | llm | StrOutputParser()

# Chain Step B: Generate checklist based on that location
checklist_prompt = ChatPromptTemplate.from_template(
    "Generate a basic background check document checklist required for a financial employee moving to {location}."
)
sequential_chain = extract_chain | (lambda location: {"location": location}) | checklist_prompt | llm | StrOutputParser()

# Execution
query = "I have an incoming hire transferring from London over to New York next month."
result = sequential_chain.invoke({"user_query": query})
print("\n[LCEL Chain Output] Localized Checklist:")
print(result)


# =====================================================================
# 2. LangChain Tools: Autonomous Tool Choice
# =====================================================================
print("\n" + "="*60 + "\n")
print("--- NATIVE LLM TOOL CALLING ---")

@tool
def check_background_status(candidate_name: str) -> str:
    """Look up the background screening verification status for a specific candidate."""
    # Mocking database logic
    records = {
        "john doe": "Status: FLAGGED. Discrepancy found in past employment dates at Barclays.",
        "jane smith": "Status: CLEAR. Fully verified."
    }
    return records.get(candidate_name.lower(), "Candidate profile not found.")

# Bind the tool directly to the model
llm_with_tools = llm.bind_tools([check_background_status])

# Query A: User asks a question requiring the tool
tool_query = "Has John Doe's background check cleared yet?"
response = llm_with_tools.invoke(tool_query)

print(f"\nUser Query: '{tool_query}'")
print("Does the LLM want to call a tool?", bool(response.tool_calls))
if response.tool_calls:
    print("Tool Selected by LLM:", response.tool_calls[0]["name"])
    print("Arguments Extracted by LLM:", response.tool_calls[0]["args"])
    
    # Simulating tool execution
    tool_output = check_background_status.invoke(response.tool_calls[0]["args"])
    print("Tool Output:", tool_output)