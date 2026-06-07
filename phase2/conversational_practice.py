import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENV_PATH = os.path.join(ROOT,"phase1",".env")
DOC_PATH = os.path.join(ROOT,"phase1","manual_rag","knowledge.txt")
load_dotenv(dotenv_path=ENV_PATH)

documents = TextLoader(DOC_PATH).load()
splitter = RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=50)
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(documents=chunks,embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k":3})

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

q_prompt = ChatPromptTemplate.from_messages([
    ("system",
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question which can be "
    "understood without the chat history. Do NOT answer the question — just "
    "reformulate it if needed, otherwise return it unchanged."),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    q_prompt,
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using ONLY the context "
     "below. If the answer is not in the context, say you don't know.\n\n"
     "{context}"),                         # retrieved docs get inserted here
    MessagesPlaceholder("chat_history"),   # prior turns, for resolving references
    ("human", "{input}"),     
])

question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id):
    # Return the history for this session, creating an empty one if it's new.
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",        # which input key holds the user's question
    history_messages_key="chat_history",  # which prompt slot to inject history into
    output_messages_key="answer",      # which output key holds the reply to save
)

session = {"configurable": {"session_id": "tanmay_session"}}

q1 = conversational_rag_chain.invoke({"input": "What is the flight time of the Zephyr X1?"}, config=session)
print(f"Q1: {q1['answer']}\n")

q2 = conversational_rag_chain.invoke(
    {"input": "What is the flight time in windy conditions?"}, 
    config=session
)

q3 = conversational_rag_chain.invoke({"input": "How long does its battery take to charge?"}, config=session)
print(f"Q3: {q3['answer']}\n")