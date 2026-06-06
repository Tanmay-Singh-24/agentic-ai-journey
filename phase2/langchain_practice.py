import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
 
DOC_PATH = ("/Users/tanmay/VSCode/agentic-ai-journey/phase1/manual_rag/knowledge.txt")
load_dotenv(dotenv_path="/Users/tanmay/VSCode/agentic-ai-journey/phase1/.env")

loader=TextLoader(DOC_PATH)
documents = loader.load()
print(f"DOCUMENTS LOADED: {len(documents)}")

splitter= RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
chunks=splitter.split_documents(documents)

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
retriever=vector_store.as_retriever(search_kwargs={"k":3})


prompt=ChatPromptTemplate.from_messages([
    ("system",
     "You are a comedian who cracks joke and annoys the user when he/she asks a question.If the answer is not in the document reply with humor."),
     ("human",
      "Context:\n{context}\n\nQuestion: {question}"),
]
)


llm=ChatGroq(model="llama-3.1-8b-instant",temperature=0.2)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain=(
    {"context": retriever|format_docs,"question":RunnablePassthrough()}
    |prompt
    | llm 
    | StrOutputParser()
)
question = "What is the flight time of the Zephyr X1?"
print(chain.invoke(question))
