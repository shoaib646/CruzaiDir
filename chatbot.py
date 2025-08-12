import streamlit as st
import os
import tempfile
import json
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Dict, Any, Optional
from docx import Document
import requests
from datetime import datetime
import pandas as pd
import io
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler

st.set_page_config(page_title="Cruz AI Chatbot", page_icon="🤖", layout="wide")

# Apply custom CSS
st.markdown("""
<style>
    .chat-bubble {
        padding: 10px 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
    }
    .chat-bubble.user {
        background-color: #e1f5fe;
        margin-left: 20%;
        border: 1px solid #b3e5fc;
    }
    .chat-bubble.assistant {
        background-color: #f5f5f5;
        margin-right: 20%;
        border: 1px solid #e0e0e0;
    }
    .chat-bubble.agent-thinking {
        background-color: #fff8e1;
        margin-right: 20%;
        border: 1px solid #ffe082;
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 0.85rem;
    }
    .chat-bubble .icon {
        margin-right: 10px;
        font-size: 1.2rem;
    }
    .tool-call {
        background-color: #f0f4f8;
        border-left: 3px solid #5c6bc0;
        padding: 8px;
        margin: 5px 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .tool-result {
        background-color: #e8f5e9;
        border-left: 3px solid #66bb6a;
        padding: 8px;
        margin: 5px 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hey There! I'm Cruz AI agent. I can help with documents, answer questions, get web information, and more. How can I help you today?",
    }]
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "agent_trace" not in st.session_state:
    st.session_state.agent_trace = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# File processing function
def process_file(files):
    documents = []
    for file in files:
        filename = file.name.lower()
        try:
            if filename.endswith('.pdf'):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                documents.extend(loader.load())
                os.remove(tmp_path)
            elif filename.endswith(('.docx', '.doc')):
                doc = Document(file)
                text = "\n".join([para.text for para in doc.paragraphs])
                DocumentObj = type("Document", (object,), {})
                doc_obj = DocumentObj()
                doc_obj.page_content = text
                doc_obj.metadata = {"source": file.name}
                documents.append(doc_obj)
            elif filename.endswith('.csv'):
                df = pd.read_csv(file)
                text = df.to_string()
                DocumentObj = type("Document", (object,), {})
                doc_obj = DocumentObj()
                doc_obj.page_content = text
                doc_obj.metadata = {"source": file.name}
                documents.append(doc_obj)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file)
                text = df.to_string()
                DocumentObj = type("Document", (object,), {})
                doc_obj = DocumentObj()
                doc_obj.page_content = text
                doc_obj.metadata = {"source": file.name}
                documents.append(doc_obj)
            elif filename.endswith(('.txt', '.json', '.md')):
                text = file.read().decode('utf-8')
                DocumentObj = type("Document", (object,), {})
                doc_obj = DocumentObj()
                doc_obj.page_content = text
                doc_obj.metadata = {"source": file.name}
                documents.append(doc_obj)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)

# Sidebar
with st.sidebar:
    st.title("🤖 Cruz AI Agent")

    with st.expander("🔑 API Keys", expanded=True):
        openai_api_key = st.text_input("OpenAI Key:", type="password")
        pinecone_api_key = st.text_input("Pinecone Key:", type="password")
        serper_api_key = st.text_input("Serper API Key (optional):", type="password", 
                                      help="For web search capability")

    # Document Manager section
    st.divider()
    st.subheader("📁 Document Manager")
    uploaded_files = st.file_uploader("Upload Documents",
                                      type=["pdf", "docx", "doc", "txt", "csv", "xlsx", "json", "md"],
                                      accept_multiple_files=True,
                                      help="Upload multiple document types for processing")

    upload_progress = st.progress(0, text="Ready for upload...")

    if st.button("🚀 Process & Index Documents", type="primary"):
        if uploaded_files and openai_api_key and pinecone_api_key:
            try:
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    api_key=openai_api_key,
                    dimensions=1536
                )
                
                # Initialize Pinecone index
                pc = Pinecone(api_key=pinecone_api_key)
                # Check if index exists, if not create it
                if "cruzai-index" not in [idx.name for idx in pc.list_indexes()]:
                    pc.create_index(
    name="cruzai-index",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ),
    dimension=1536,
    metric="cosine"
)
                
                index = pc.Index("cruzai-index")
                vector_store = PineconeVectorStore(embedding=embeddings, index=index)

                total_files = len(uploaded_files)
                processed_docs = []

                for i, file in enumerate(uploaded_files):
                    upload_progress.progress((i + 1) / total_files,
                                             text=f"Processing {file.name}...")
                    processed_docs.extend(process_file([file]))

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                )
                splits = text_splitter.split_documents(processed_docs)

                with st.spinner("📄 Uploading to vector database..."):
                    vector_store.add_documents(splits)

                st.session_state.uploaded_files = uploaded_files
                upload_progress.empty()
                st.success(f"✅ Processed {len(splits)} chunks from {total_files} files!")

            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
        else:
            st.warning("⚠️ Please upload documents and enter API keys first")

    # Uploaded files list
    if st.session_state.uploaded_files:
        st.subheader("Uploaded Files")
        for file in st.session_state.uploaded_files:
            col1, col2 = st.columns([0.8, 0.2])
            col1.caption(file.name)
            col2.button(
                "❌",
                key=f"remove_{file.name}",
                on_click=lambda f=file: st.session_state.uploaded_files.remove(f)
            )

    st.divider()
    
    # Advanced options
    with st.expander("⚙️ Agent Settings", expanded=False):
        agent_model = st.selectbox(
            "Agent Model",
            ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=2
        )
        st.caption("More powerful models are more capable but slower and more expensive")
        
        show_reasoning = st.toggle("Show Agent Reasoning", value=False)
        if show_reasoning:
            st.caption("This will display the agent's step-by-step thinking process")

    if st.button("🧹 Delete All Indexed Data", type="secondary"):
        if pinecone_api_key:
            try:
                pc = Pinecone(api_key=pinecone_api_key)
                if "cruzai-index" in [idx.name for idx in pc.list_indexes()]:
                    index = pc.Index("cruzai-index")
                    with st.spinner("Deleting all vector data from Pinecone index..."):
                        index.delete(delete_all=True)
                    st.success("✅ All vectors deleted from the index.")
                else:
                    st.warning("No index exists to delete.")
            except Exception as e:
                st.error(f"❌ Failed to delete vectors: {str(e)}")
        else:
            st.warning("⚠️ Please provide your Pinecone API key.")

    if st.button("🧼 Clear Chat History", type="secondary"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "🔃 Chat history cleared! How can I help you?",
        }]
        st.session_state.agent_trace = []
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.rerun()

# Main chat UI
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
        <h1 style='text-align: center; font-size: 1.5rem; font-weight: 700; 
        margin-bottom: 1rem; width: 100%; display: block;'>
            Cruz AI Agent - Your Intelligent Assistant
        </h1>
    """, unsafe_allow_html=True)

    # Chat messages display
    for message in st.session_state.messages:
        role_class = "user" if message["role"] == "user" else "assistant"
        icon = "👤" if message["role"] == "user" else "🤖"
        st.markdown(
            f"""<div class='chat-bubble {role_class}'>
                <span class='icon'>{icon}</span>
                <div>{message['content']}</div>
            </div>""",
            unsafe_allow_html=True
        )

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar first!")
            st.stop()

        try:
            llm = init_chat_model(agent_model if 'agent_model' in locals() else "gpt-3.5-turbo", 
                                model_provider="openai", 
                                api_key=openai_api_key)
            
            # Define agent tools
            tools = []
            
            # Retrieval tool (if Pinecone is configured)
            if pinecone_api_key:
                try:
                    embeddings = OpenAIEmbeddings(
                        model="text-embedding-3-large",
                        api_key=openai_api_key,
                        dimensions=1536
                    )
                    pc = Pinecone(api_key=pinecone_api_key)
                    
                    # Check if index exists
                    if "cruzai-index" in [idx.name for idx in pc.list_indexes()]:
                        index = pc.Index("cruzai-index")
                        vector_store = PineconeVectorStore(embedding=embeddings, index=index)
                        
                        @tool()
                        def retrieve(query: str) -> str:
                            """Search through indexed documents for relevant information.
                            Use this when asked about specific document content or when you need to reference uploaded documents."""
                            try:
                                retrieved_docs = vector_store.similarity_search(query, k=3)
                                serialized = "\n\n".join(
                                    (f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}")
                                    for doc in retrieved_docs
                                )
                                return serialized if serialized else "No relevant documents found."
                            except Exception as e:
                                return f"Error retrieving documents: {str(e)}"
                        
                        tools.append(retrieve)
                except Exception as e:
                    st.error(f"Error setting up retrieval tool: {e}")
            
            # Web search tool (if Serper API key is provided)
            if serper_api_key:
                @tool()
                def web_search(query: str) -> str:
                    """Search the web for real-time information. 
                    Use this for current events, facts, or information not in the user's documents."""
                    try:
                        headers = {
                            'X-API-KEY': serper_api_key,
                            'Content-Type': 'application/json'
                        }
                        payload = json.dumps({"q": query})
                        response = requests.post(
                            'https://google.serper.dev/search',
                            headers=headers,
                            data=payload
                        )
                        response_data = response.json()
                        
                        results = []
                        if 'organic' in response_data:
                            for item in response_data['organic'][:3]:
                                results.append(f"Title: {item.get('title', 'No title')}")
                                results.append(f"Link: {item.get('link', 'No link')}")
                                results.append(f"Snippet: {item.get('snippet', 'No snippet')}")
                                results.append("---")
                        
                        return "\n".join(results) if results else "No search results found."
                    except Exception as e:
                        return f"Error searching the web: {str(e)}"
                
                tools.append(web_search)
            
            # Current date and time tool
            @tool()
            def get_current_datetime() -> str:
                """Get the current date and time. Use this when asked about current time, date, day of week, etc."""
                now = datetime.now()
                return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}\nDay of week: {now.strftime('%A')}"
            
            tools.append(get_current_datetime)
            
            # Data analysis tool
            @tool()
            def analyze_data(data: str, analysis_type: str) -> str:
                """Analyze tabular data provided as CSV string.
                data should be a string in CSV format.
                analysis_type can be 'summary', 'statistics', or 'chart'."""
                try:
                    df = pd.read_csv(io.StringIO(data))
                    
                    if analysis_type.lower() == 'summary':
                        return f"Data shape: {df.shape}\nColumns: {', '.join(df.columns.tolist())}\nSample:\n{df.head(3).to_string()}"
                    
                    elif analysis_type.lower() == 'statistics':
                        return df.describe().to_string()
                    
                    elif analysis_type.lower() == 'chart':
                        # Create a simple chart (this will just return a description)
                        return f"Chart would display data with columns: {', '.join(df.columns.tolist())}\nTo create an actual chart, please use a plotting library in a separate step."
                    
                    else:
                        return "Invalid analysis type. Choose 'summary', 'statistics', or 'chart'."
                    
                except Exception as e:
                    return f"Error analyzing data: {str(e)}"
            
            tools.append(analyze_data)
            
            # Set up the agent
            memory = MemorySaver()
            
            # Create conversation history from session state messages
            conversation_history = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    conversation_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    conversation_history.append(AIMessage(content=msg["content"]))
            
            # Initialize the agent
            agent = create_react_agent(
                llm,
                tools,
                prompt="""You are Cruz AI, an intelligent agent designed to assist users with various tasks.
                You have access to several tools:
                - retrieve: Search through indexed documents
                - web_search: Get information from the web (if available)
                - get_current_datetime: Provide the current date and time
                - analyze_data: Analyze data provided in CSV format
                
                Always be helpful, accurate, and concise. If you don't know something, say so rather than making up information.
                If a question relates to documents, use the retrieve tool.
                If a question is about current events or facts not in documents, use web_search if available.
                First try to understand what the user is asking for, then select the appropriate tool to help answer their question.
                """,
                checkpointer=memory,
            )
            
            # Add the new user message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Initialize trace collection
            trace_events = []
            
            # Display thinking status
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown(
                """<div class='chat-bubble agent-thinking'>
                    <span class='icon'>🧠</span>
                    <div>Thinking...</div>
                </div>""",
                unsafe_allow_html=True
            )
            
            # Prepare conversation history for the agent
            current_messages = conversation_history + [HumanMessage(content=prompt)]
            
            
            
            # Execute the agent
            config = {
                "configurable": {
                    "thread_id": st.session_state.conversation_id
                },
            }
            
            response = agent.invoke(
                {"messages": current_messages},
                config=config
            )
            
            # Store the trace
            st.session_state.agent_trace.append({
                "query": prompt,
                "events": trace_events,
                "response": response
            })
            
            # Clear thinking status and add assistant response
            thinking_placeholder.empty()
            
            # Extract the final assistant message
            final_message = response["messages"][-1].content
            
            # Add the response to session state
            st.session_state.messages.append({"role": "assistant", "content": final_message})
            
            # Rerun to update the UI
            st.rerun()

        except Exception as e:
            st.error(f"Error processing request: {e}")

with col2:
    if 'show_reasoning' in locals() and show_reasoning and st.session_state.agent_trace:
        st.subheader("Agent Reasoning")
        
        # Display the most recent trace
        latest_trace = st.session_state.agent_trace[-1]
        
        for event in latest_trace["events"]:
            if event["type"] == "tool_start":
                st.markdown(f"""<div class='tool-call'>
                    <strong>🔧 Tool Call:</strong> {event["name"]}
                    <br/><strong>Input:</strong> {event["input"]}
                    </div>""", 
                    unsafe_allow_html=True)
            
            elif event["type"] == "tool_end":
                st.markdown(f"""<div class='tool-result'>
                    <strong>🔄 Tool Result:</strong>
                    <br/>{event["output"]}
                    </div>""", 
                    unsafe_allow_html=True)
            
            elif event["type"] == "agent_action":
                st.markdown(f"""<div class='agent-action'>
                    <strong>🤔 Agent Action:</strong>
                    <br/>{event["action"]}
                    </div>""", 
                    unsafe_allow_html=True)
