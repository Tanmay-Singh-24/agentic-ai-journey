from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(dotenv_path="../phase1/.env")

llm = ChatGroq(model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI engineering tutor. Answer questions about AI and ML only."),
    ("human", "{question}")
])

chain = prompt | llm

response = chain.invoke({"question": "What is Retrieval Augmented Generation in one sentence?"})
print(response.content)