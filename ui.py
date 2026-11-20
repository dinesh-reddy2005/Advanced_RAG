
import streamlit as st
import time
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from src.rag_pipeline import rag_chain
from src.config import retriever


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Advanced RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

    /* Main app background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    /* Main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border: 1px solid #30363d;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 12px;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        border-radius: 12px;
    }

    /* Buttons */
    .stButton button {
        border-radius: 10px;
        min-height: 40px;
    }

    /* Headings */
    h1, h2, h3 {
        color: #ffffff;
    }

    /* Info cards */
    .info-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 10px;
    }

    /* Caption */
    .small-text {
        color: #8b949e;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================================
# SESSION STATE
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("# 🤖 Advanced RAG")

    st.caption("AI-powered document assistant")

    st.divider()

    # New chat
    if st.button("➕ New Chat", use_container_width=True):

        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.chat_started = False

        st.rerun()

    st.markdown("### ⚙️ Settings")

    show_sources = st.toggle(
        "Show retrieved documents",
        value=True
    )

    show_processing_time = st.toggle(
        "Show response time",
        value=True
    )

    show_chat_history = st.toggle(
        "Show conversation history",
        value=True
    )

    st.divider()

    st.markdown("### 🧠 RAG Information")

    st.markdown("""
    **Architecture**

    - FAISS Vector Database
    - HuggingFace Embeddings
    - Groq LLM
    - LangChain
    - Contextual Retrieval
    - Intent Classification
    """)

    st.divider()

    st.markdown("### 📄 Knowledge Base")

    st.success("Vector database connected")

    st.caption("Source: company_policy.pdf")

    st.caption("Embedding: all-MiniLM-L6-v2")

    st.caption("Retriever: Similarity Search")

    st.caption("Top-K: 3 (backend setting)")

    st.divider()

    # Download conversation
    if st.session_state.messages:

        conversation_text = ""

        for message in st.session_state.messages:

            role = message["role"].upper()

            conversation_text += (
                f"\n\n{role}:\n{message['content']}\n"
            )

        st.download_button(
            label="📥 Download Conversation",
            data=conversation_text,
            file_name="rag_conversation.txt",
            mime="text/plain",
            use_container_width=True
        )

    if st.button("🗑️ Clear Conversation", use_container_width=True):

        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.chat_started = False

        st.rerun()


# ==========================================================
# MAIN HEADER
# ==========================================================

st.markdown("# 🤖 Advanced RAG Chatbot")

st.markdown(
    """
    Ask questions about your documents.
    The chatbot uses retrieval-augmented generation to provide
    context-based answers.
    """
)

st.markdown("---")


# ==========================================================
# WELCOME SCREEN
# ==========================================================

if not st.session_state.messages:

    st.markdown("## 👋 Welcome!")

    st.markdown(
        "I'm your AI document assistant. "
        "Ask me anything about the uploaded knowledge base."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="info-card">

        ### 📚 Document QA

        Ask questions directly from company policy documents.

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-card">

        ### 🔄 Contextual Chat

        Ask follow-up questions without repeating context.

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="info-card">

        ### 🛡️ Grounded Responses

        The document-answering prompt restricts responses
        to retrieved context.

        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 💡 Try asking")

    example_questions = [
        "What is the company leave policy?",
        "What are the different types of leave?",
        "What was my previous question?",
        "Hello, how are you?"
    ]

    cols = st.columns(2)

    for index, question in enumerate(example_questions):

        with cols[index % 2]:

            if st.button(
                question,
                key=f"example_{index}",
                use_container_width=True
            ):

                st.session_state.pending_question = question
                st.rerun()


# ==========================================================
# DISPLAY EXISTING MESSAGES
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display source documents if available
        if (
            message["role"] == "assistant"
            and message.get("sources")
            and show_sources
        ):

            with st.expander("📚 Retrieved document chunks"):

                for i, source in enumerate(message["sources"]):

                    st.markdown(f"**Source {i + 1}**")

                    st.caption(
                        f"Page: {source.get('page', 'Unknown')}"
                    )

                    st.write(source["content"])

                    st.divider()

        # Display response time
        if (
            message["role"] == "assistant"
            and message.get("response_time")
            and show_processing_time
        ):

            st.caption(
                f"⏱️ Response time: "
                f"{message['response_time']:.2f} seconds"
            )


# ==========================================================
# USER INPUT
# ==========================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


# Handle example questions
if "pending_question" in st.session_state:

    question = st.session_state.pending_question

    del st.session_state.pending_question


# ==========================================================
# CHAT PROCESSING
# ==========================================================

if question:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("🧠 Analyzing your question..."):

            start_time = time.time()

            try:

                # Invoke existing RAG chain
                answer = rag_chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history[-10:]
                })

                response_time = time.time() - start_time

                # Display answer
                st.markdown(answer)

                # Retrieve source chunks for display
                source_documents = []

                if show_sources:

                    try:

                        docs = retriever.invoke(question)

                        for doc in docs:

                            source_documents.append({
                                "content": doc.page_content,
                                "page": doc.metadata.get(
                                    "page",
                                    "Unknown"
                                ),
                                "source": doc.metadata.get(
                                    "source",
                                    "Unknown"
                                )
                            })

                    except Exception as source_error:

                        st.warning(
                            "Unable to retrieve source previews."
                        )


                # Display sources
                if source_documents and show_sources:

                    with st.expander(
                        "📚 View retrieved document chunks"
                    ):

                        for index, source in enumerate(
                            source_documents
                        ):

                            st.markdown(
                                f"### Source {index + 1}"
                            )

                            st.caption(
                                f"Page: {source['page']}"
                            )

                            st.write(source["content"])

                            st.divider()


                # Processing time
                if show_processing_time:

                    st.caption(
                        f"⏱️ Response generated in "
                        f"{response_time:.2f} seconds"
                    )


                # Store assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_documents,
                    "response_time": response_time
                })


                # Store conversation history
                st.session_state.chat_history.append(
                    HumanMessage(content=question)
                )

                st.session_state.chat_history.append(
                    AIMessage(content=answer)
                )


            except Exception as e:

                st.error(
                    f"An error occurred while generating the response: {e}"
                )


    # Rerun to refresh UI
    st.rerun()
