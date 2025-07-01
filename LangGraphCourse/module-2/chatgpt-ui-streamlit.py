from openai import OpenAI
import streamlit as st
import os

from my_config import MyConfig

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

import uuid


my_config = MyConfig()

langfuse = Langfuse(
    public_key = my_config.LANGFUSE_PUBLIC_KEY,
    secret_key = my_config.LANGFUSE_SECRET_KEY,
    host = my_config.LANGFUSE_HOST
)
langfuse_handler1 = CallbackHandler()

# pull file if it doesn't exist and connect to local db
import sqlite3

db_path = "state_db/example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)


# Here is our checkpointer 
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver(conn)


from langgraph.checkpoint.memory import MemorySaver
#  memory =  MemorySaver()


from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

from langgraph.graph import END
from langgraph.graph import MessagesState

model = ChatOpenAI(model="gpt-4o",temperature=0)

class State(MessagesState):
    summary: str

# Define the logic to call the model
def call_model(state: State):
    
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        messages = state["messages"]
    
    response = model.invoke(messages)
    return {"messages": response}

def summarize_conversation(state: State):
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# Determine whether to end or summarize the conversation
def should_continue(state: State):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END

from IPython.display import Image, display
from langgraph.graph import StateGraph, START

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
graph = workflow.compile(checkpointer=memory)

# give title to the page
st.title('OpenAI ChatGPT')

# initialize session variables at the start once
if 'model' not in st.session_state:
    #st.session_state['model'] = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    st.session_state['model'] = graph
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = str(uuid.uuid4())

# create sidebar to adjust parameters
st.sidebar.title('Model Parameters')
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
max_tokens = st.sidebar.slider('Max Tokens', min_value=1, max_value=4096, value=256)

# update the interface with the previous messages
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# create the chat interface
if prompt := st.chat_input("Enter your query"):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    # get response from the model
    with st.chat_message('assistant'):
        client = st.session_state['model']
        config = {"configurable": {"thread_id": st.session_state['thread_id']}, "callbacks": [langfuse_handler1]}
        # pass only the last message to the graph
        stream = graph.invoke({"messages": [{"role": "user", "content": prompt}]}, config) 
        # print(stream['messages'][-1].content) 
        response_str = stream['messages'][-1].content 
        response = st.write(stream['messages'][-1].content)
        
    st.session_state['messages'].append({"role": "assistant", "content": response_str})
    #print(st.session_state['messages'])

    # handle message overflow based on the model sizes