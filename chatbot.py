import streamlit as st
import os
import tempfile
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing import List
from docx import Document

st.set_page_config(page_title="Cruzai RAG", page_icon="📝")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hey There! I'm Cruz AI assistant. How can I help you?",
    }]
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

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
    st.title("🔧 Configuration")

    with st.expander("🔑 API Keys", expanded=True):
        openai_api_key = st.text_input("OpenAI Key:", type="password")
        pinecone_api_key = st.text_input("Pinecone Key:", type="password")

    # Document Manager section
    st.divider()
    st.subheader("📁 Document Manager")
    uploaded_files = st.file_uploader("Upload Documents",
                                      type=["pdf", "docx", "doc"],
                                      accept_multiple_files=True,
                                      help="Upload multiple PDF or Word documents")

    upload_progress = st.progress(0, text="Ready for upload...")

    if st.button("🚀 Process & Index Documents", type="primary"):
        if uploaded_files and openai_api_key and pinecone_api_key:
            try:
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    api_key=openai_api_key,
                    dimensions=1536
                )
                index = Pinecone(api_key=pinecone_api_key).Index("myindex")
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

    # Uploaded files list without nested expander
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

    if st.button("🧹 Delete All Indexed Data", type="secondary"):
        if pinecone_api_key:
            try:
                index = Pinecone(api_key=pinecone_api_key).Index("myindex")
                with st.spinner("Deleting all vector data from Pinecone index..."):
                    index.delete(delete_all=True)
                st.success("✅ All vectors deleted from the index.")
            except Exception as e:
                st.error(f"❌ Failed to delete vectors: {str(e)}")
        else:
            st.warning("⚠️ Please provide your Pinecone API key.")

    if st.button("🩹 Clear Chat History", type="secondary"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "🔃 Chat history cleared! How can I help you?",
        }]
        st.rerun()

# Main chat UI
st.markdown("""
    <h1 style='text-align: center; font-size: 1.5rem; font-weight: 700; 
    margin-bottom: 1rem; width: 100%; display: block;'>
        Custom RAG! Chat with your Docs! Don't worry, I don't save any data.
    </h1>
""", unsafe_allow_html=True)

# Chat messages display
for message in st.session_state.messages:
    role_class = "user" if message["role"] == "user" else "assistant"
    icon = "👤" if message["role"] == "user" else "🧠"
    st.markdown(
        f"""<div class='chat-bubble {role_class}'>
            <span class='icon'>{icon}</span>
            <div>{message['content']}</div>
        </div>""",
        unsafe_allow_html=True
    )

# Chat input
if prompt := st.chat_input("Ask me anything about your document..."):
    if not openai_api_key or not pinecone_api_key:
        st.warning("Please enter both API keys in the sidebar first!")
        st.stop()

    try:
        llm = init_chat_model("gpt-3.5-turbo", model_provider="openai", api_key=openai_api_key)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=openai_api_key,
            dimensions=1536
        )
        index = Pinecone(api_key=pinecone_api_key).Index("myindex")
        vector_store = PineconeVectorStore(embedding=embeddings, index=index)

        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = vector_store.similarity_search(query, k=3)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        memory = MemorySaver()
        agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
        config = {"configurable": {"thread_id": "1a"}}
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            response = agent_executor.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config
            )["messages"][-1].content

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    except Exception as e:
        st.error(f"Error processing request: {e}")
