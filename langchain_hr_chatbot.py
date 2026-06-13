"""
This script demonstrates how to use Langchain to create an HR chatbot that evaluates candidates for a Managing Director role in Risk Management at a Wall Street Bank.

You'll need to install the Langchain associated Gemini or OpenAI library and set your GEMINI_API_KEY / OPENAI_API_KEY environment variable before running this script.

To install the Gemini API client library, run:
pip install -U langchain-core langchain-google-genai pydantic
pip install -U langchain-openai

To set your GEMINI_API_KEY environment variable, you can do it in your terminal like this:
export GEMINI_API_KEY="your_actual_api_key_here"
OR
export OPENAI_API_KEY="your_actual_api_key_here"

To run the script, simply execute:
python langchain_hr_chatbot.py

"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Ensure your GEMINI_API_KEY environment variable is set
# os.environ["GEMINI_API_KEY"] = "your_actual_api_key_here"

# Define the desired output structure using Pydantic
class CandidateEvaluation(BaseModel):
    is_qualified: bool = Field(
        description="True if the candidate meets the strict criteria for a Managing Director role, False otherwise."
    )
    reasoning: str = Field(
        description="A highly analytical, objective breakdown of why the candidate is or is not qualified, referencing regulatory standards and experience depth."
    )

# Initialize the Gemini LLM via LangChain

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
# llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# Bind the Pydantic schema to the model to enforce structured output
structured_llm = llm.with_structured_output(CandidateEvaluation)

# Construct the prompt template using system and user roles
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
You are a Senior Executive Talent Recruiter specializing in risk and compliance for global tier-1 investment banks. 
Your tone must be highly analytical, direct, and objective. 
Do not use conversational fluff, pleasantries, or exclamation marks. 
Evaluate candidates ruthlessly against strict regulatory standpoints (SEC, FINRA, Dodd-Frank, MiFID II).
"""),
    ("user", """
Analyze this candidate's resume and tell me if they are qualified for a 
Managing Director role in Risk Management at a Wall Street Bank.
Candidate Resume:
{resume_data}
""")
])

# Define the raw mock-up resume data
RAW_RESUME = """
SARAH JENKINS
Experience:
- Senior Associate, Quant Trading at Barclays (2022 - Present)
  * Built algorithmic trading models for European equities.
  * Ensured compliance with MiFID II regulations.
- Analyst, Risk Management at HSBC (2020 - 2022)
  * Monitored daily VaR limits and liquidity risks.
Education:
- M.S. in Financial Engineering, Imperial College London
Certifications:
- CFA Level II Candidate
"""

# Chain the components together and invoke
chain = prompt_template | structured_llm

# The response returned here is a fully parsed CandidateEvaluation Pydantic object
evaluation: CandidateEvaluation = chain.invoke({"resume_data": RAW_RESUME})

print("\nLangChain Structured Output (Pydantic Object):\n")
print(evaluation)
print(f"Is Qualified: {evaluation.is_qualified}")
print(f"Reasoning:\n{evaluation.reasoning}")

# If you need it as raw JSON or a dictionary:
# print(evaluation.model_dump_json(indent=2))