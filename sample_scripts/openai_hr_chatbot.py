"""
This script demonstrates how to use the OpenAI API to create an HR chatbot that evaluates candidates for a Managing Director role in Risk Management at a Wall Street Bank.

You'll need to install the OpenAI API client library and set your OpenAI_API_KEY environment variable before running this script.

To install the OpenAI API client library, run:
pip install openai pydantic

To set your OPENAI_API_KEY environment variable, you can do it in your terminal like this:
export OPENAI_API_KEY="your_actual_api_key_here"

To run the script, simply execute:
python openai_hr_chatbot.py
"""

import os
from openai import OpenAI
from pydantic import BaseModel, Field

# Ensure your OpenAI_API_KEY environment variable is set
# os.environ["OpenAI_API_KEY"] = "your_actual_api_key_here"

# Initialize the OpenAI client
client = OpenAI()
MODEL_ID = "gpt-4o"

# System prompt
system_instruction = """
You are a Senior Executive Talent Recruiter specializing in risk and compliance for global tier-1 investment banks. 
Your tone must be highly analytical, direct, and objective. 
Do not use conversational fluff, pleasantries, or exclamation marks. 
Evaluate candidates ruthlessly against strict regulatory standpoints (SEC, FINRA, Dodd-Frank, MiFID II).
"""

# Raw mock-up resume data
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

# Setup the user prompt asking for candidate assessment
user_prompt = f"""
Analyze this candidate's resume and tell me if they are qualified for a 
Managing Director role in Risk Management at a Wall Street Bank.
Candidate Resume:
{RAW_RESUME}
"""

# response = client.models.generate_content(
#     model=MODEL_ID,
#     contents=user_prompt,
#     config=types.GenerateContentConfig(
#         system_instruction=system_instruction,
#         temperature=0.2  # Low temperature keeps the evaluation professional and focused
#     )
# )
# print("\nLLM output:\n")
# print(response.text)



# Define the desired output structure using Pydantic
class CandidateEvaluation(BaseModel):
    is_qualified: bool = Field(
        description="True if the candidate meets the strict criteria for a Managing Director role, False otherwise."
    )
    reasoning: str = Field(
        description="A highly analytical, objective breakdown of why the candidate is or is not qualified, referencing regulatory standards and experience depth."
    )


response = client.beta.chat.completions.parse(
    model="gpt-4o",  # Or your chosen OpenAI model ID
    temperature=0.2, # Keeps the evaluation focused
    messages=[
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ],
    response_format=CandidateEvaluation,
)

# Accessing the parsed object directly
evaluation_result = response.choices[0].message.parsed
print(evaluation_result)