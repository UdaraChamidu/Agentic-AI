# creating coustom tools 
# add tool
# sub
# multiply

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

llm = LLM(
    model="gpt-3.5-turbo")

@tool("add_tool")
def add_tool(a: int, b: int) -> int:
    """Adds two numbers and returns the result."""
    return a + b

@tool("subtract_tool")
def sub_tool(a: int, b: int) -> int:
    """Subtracts two numbers and return the results"""
    return a - b

@tool("mul_tool")
def mul_tool(a: int, b: int) -> int:
    """Multiplies two numbers and returns the result."""
    return a * b


math_agent = Agent(
    role="Math Assistant",
    goal = "solve the problems by performing arithmetic operations. Calculate the result of {question}",
    backstory = "You are a skillful mathematician can perform addition, subtraction and multiplication.",
    tools=[add_tool, sub_tool, mul_tool],
    llm=llm
)

task = Task(
    description = "Calculate the result of {question}.",
    expected_output = "A single number as the final result",
    agent = math_agent
)

crew = Crew(
    agents=[math_agent],  # List of agents in the crew
    tasks=[task],
    verbose=True
)

result = crew.kickoff(inputs={"question": "What is 5 + 3*2?"})
print(result)
