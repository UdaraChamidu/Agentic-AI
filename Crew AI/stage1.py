import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

llm = LLM(
    model="gpt-3.5-turbo")

# Define the agent and task for the crew
qa_agent = Agent(
    role="question and answering agent",
    goal = "provide accurate, concise answers to user questions",
    backstory = "You are a helpful assistant trained to answer questions.",
    tools = [],
    llm=llm,
)

# Define a task for the agent
qa_task = Task(
    description = "Answer the user's question",
    expected_output = "A concise answer to the question",
    agent = qa_agent
)

# Create a crew with the agent and task
crew = Crew(
    agents=[qa_agent],  # List of agents in the crew
    tasks=[qa_task],
    verbose=True
)

result = crew.kickoff(inputs={"question": "What is the capital of France?"})
print(result)