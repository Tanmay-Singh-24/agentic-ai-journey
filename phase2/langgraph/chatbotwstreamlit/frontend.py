import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])
user_input = st.chat_input('Enter text')
if user_input:
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)
    config = {"configurable": {"thread_id": "user"}}
    response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config)
    assistant_message = response["messages"][-1].content
    st.session_state['message_history'].append({'role':'assistant','content':assistant_message})
    with st.chat_message('assistant'):
        st.text(assistant_message)