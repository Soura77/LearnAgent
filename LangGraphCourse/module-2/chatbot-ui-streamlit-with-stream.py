from openai import OpenAI
import streamlit as st
import os
import asyncio
from my_config import MyConfig

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import uuid

from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition

my_config = MyConfig()

langfuse = Langfuse(
    public_key = my_config.LANGFUSE_PUBLIC_KEY,
    secret_key = my_config.LANGFUSE_SECRET_KEY,
    host = my_config.LANGFUSE_HOST
)
langfuse_handler1 = CallbackHandler()

# pull file if it doesn't exist and connect to local db
# import sqlite3

db_path = "state_db/example.db"
# conn = sqlite3.connect(db_path, check_same_thread=False)


# Here is our checkpointer 
#  from langgraph.checkpoint.sqlite import SqliteSaver

#  memory = SqliteSaver(conn)


def get_async_memory(db_path):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return AsyncSqliteSaver(db_path)

# memory = get_async_memory(db_path)

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, ToolMessage,AIMessage

from langgraph.graph import END
from langgraph.graph import MessagesState

# Initialize the LLM and tools
llm = ChatOpenAI(model="gpt-4o",temperature=0)
tools= [TavilySearchResults(max_results=3)]

# Bind the tools to the model
model = llm.bind_tools(tools, parallel_tool_calls=False)


class State(MessagesState):
    summary: str

# Define the logic to call the model
def call_model(state: State):
    # Remove tool calls from the messages and only keep human and AI messages without tool call response
    prompt_messages = state["messages"]
    
    # Get summary if it exists
    summary = state.get("summary", "")
    system_prompt_default = "You are a helpful assistant. Answer the user's questions to the best of your ability. Ther are multiple tools available to you, including a web search tool,which can help you with any topic related to current affairs and other general information. Use the tools when necessary to answer the user's question.\n\n"
    # If there is summary, then we add it
    if summary:
        
        # Add summary to system message
        system_message = f"{system_prompt_default}Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        
        messages = [SystemMessage(content=system_message)] + prompt_messages
    
    else:
        system_message = system_prompt_default
        messages = [SystemMessage(content=system_message)] + prompt_messages
    
    response = model.invoke(messages)
    return {"messages": response}

def summarize_conversation(state: State):
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"\n\nThis is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "\n\nCreate a summary of the conversation above:"

    # Add prompt to our history
    messages = [message for message in state["messages"] if message.type == "human" or (message.type == "ai" and len(message.tool_calls) == 0)] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages 
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# Determine whether to end or summarize the conversation
def should_continue(state: State):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # Otherwise we check if we should tool call
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    return END

from IPython.display import Image, display
from langgraph.graph import StateGraph, START

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node("tools", ToolNode(tools))
# workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("tools", "conversation")
# workflow.add_edge("summarize_conversation", END)

    

# give title to the page
st.title('My Chatbot')

# initialize session variables at the start once
if 'model' not in st.session_state:
    #st.session_state['model'] = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    st.session_state['model'] = workflow
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
        input_message = {"role": "user", "content": prompt}
        node_to_stream = "conversation" 
        model = st.session_state['model'] 

        # Streaming generator
        def stream_response():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_chunks = []
            async def fetch():
                async with AsyncSqliteSaver.from_conn_string(db_path) as memory:
                    graph = model.compile(checkpointer=memory)
                    async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
                        if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
                            data = event["data"]
                            yield data["chunk"].content
            gen = fetch()
            try:
                while True:
                    chunk = loop.run_until_complete(gen.__anext__())
                    response_chunks.append(chunk)
                    yield chunk
            except StopAsyncIteration:
                pass
            finally:
                loop.close()
            # Save the full response for session state
            st.session_state['last_response'] = "".join(response_chunks)

        # Display streaming output
        response_str = st.write_stream(stream_response())
        
    st.session_state['messages'].append({"role": "assistant", "content": response_str})
    #print(st.session_state['messages'])