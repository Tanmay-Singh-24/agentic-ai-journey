from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

load_dotenv(dotenv_path="../phase1/.env")
@tool
def get_word_count(text: str) -> int:
    """Count the number of words in a given text."""
    return len(text.split())

@tool
def num_multiply(numbers: str) -> int:
    """Multiply two numbers. Input should be two numbers separated by a comma, e.g. '7,8'"""
    a, b = numbers.split(",")
    return int(a.strip()) * int(b.strip())

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
search = DuckDuckGoSearchRun()
tools = [search,get_word_count,num_multiply]

prompt = PromptTemplate.from_template("""Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}""")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What is 7 multiplied by 8? Use the num_multiply tool."})
print(result["output"])