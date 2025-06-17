# 2 agents
# 1 for research and other for write data in to a text file.

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import PDFSearchTool

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

llm = LLM(
    model="gpt-3.5-turbo")

pdf_rag_tool = PDFSearchTool(pdf="E:\crewAI RAG\machine learning.pdf")

researcher = Agent(
    role="Information Research Assistant",
    goal="Extract relevant information to support user query: {query}",
    backstory="You are skilled in analysing documents and finding key insights quickly and efficiently.",
    tools=[pdf_rag_tool]
)

writer = Agent(
    role="Content Generator Assistant",
    goal="create clear, engaging and well structured written related to query: {query}.",
    backstory="You are passionate about turning ideas and facts into compiling, easy-to-read content.",
    tools=[],
)

research_task = Task(
    description="Analyze the provided document and extract key facts, statistics, and insights related to the user's query.",
    expected_output="A structured list or summary of important finding, categorical by subject or theme.",
    agent=researcher
)
writer_task = Task(
    description="Use the extracted research findings to create a well-written report that addresses the user's query.",
    expected_output="A polished, three-sections report in markdown format with an introduction, main content, and conclusion.",
    output_file="report.md",
    agent=writer
)

crew = Crew(
    agents=[researcher, writer],  # List of agents in the crew
    tasks=[research_task, writer_task],
    verbose=True
)

result = crew.kickoff(
    inputs={"query": "What is machine learning?"})
print(result)

