import streamlit as st
import os
import json
import time
from Main import process_file, create_qa_system

# Set page config
st.set_page_config(
    page_title="Chat With Your Data",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="💬"
)

# Dark theme styling
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #2a2a2a;
        color: white;
    }
    .stButton>button {
        background-color: #444;
        color: white;
        border-radius: 8px;
    }
    .stDownloadButton>button {
        background-color: #4a90e2;
        color: white;
        border-radius: 8px;
    }
    .message-container {
        display: inline-block;
        max-width: 70%;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 8px 0;
        word-wrap: break-word;
    }
    .user-message {
        background-color: #0078D4;
        color: white;
        text-align: right;
    }
    .ai-message {
        background-color: #333;
        color: white;
    }
    .typing-animation span {
        height: 8px;
        width: 8px;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 50%;
        display: inline-block;
        margin: 0 2px;
        animation: bounce 1.5s infinite ease-in-out;
    }
    @keyframes bounce {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-5px); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("📄 Chat With Your Data")

# File Upload Section
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

uploaded_file = st.file_uploader(
    "📂 Upload your document",
    type=["pdf", "txt", "csv"],
    key="file_uploader"
)

if uploaded_file:
    file_extension = os.path.splitext(uploaded_file.name)[-1]
    temp_path = os.path.join(TEMP_DIR, f"uploaded_file{file_extension}")

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing your document..."):
        vectorstore = process_file(temp_path)
        st.session_state["qa_chain"] = create_qa_system(vectorstore)
        st.session_state["chat_started"] = True
    st.success("✅ Document processed! Ask me anything.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_question" not in st.session_state:
    st.session_state.last_question = None

# Chat UI
if st.session_state.get("chat_started"):
    st.subheader("💬 Conversation")

    # Display chat messages
    for role, message in st.session_state.chat_history:
        class_name = "user-message" if role == "You" else "ai-message"
        st.markdown(
            f'<div class="message-container {class_name}"><b>{role}:</b> {message}</div>',
            unsafe_allow_html=True
        )

    # Input area - key changes on each submission to force reset
    input_key = f"user_input_{len(st.session_state.chat_history)}"
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question...",
            key=input_key,
            label_visibility="collapsed",
            placeholder="Type your question here..."
        )
    with col2:
        send_button = st.button("📤", key=f"send_{len(st.session_state.chat_history)}")

    # Handle question submission
    if send_button and user_input:
        # Store the question before clearing
        st.session_state.last_question = user_input
        st.rerun()

# Process question after rerun (avoids double-processing)
if "last_question" in st.session_state and st.session_state.last_question:
    question = st.session_state.last_question
    st.session_state.last_question = None
    
    # Add user message to history
    st.session_state.chat_history.append(("You", question))
    
    # Show typing indicator
    with st.empty():
        st.markdown(
            '<div class="message-container ai-message">'
            '<b>AI:</b> '
            '<div class="typing-animation">'
            '<span></span><span></span><span></span>'
            '</div></div>',
            unsafe_allow_html=True
        )
        time.sleep(1.5)
        
        # Get AI response
        response = st.session_state["qa_chain"].run(question)
    
    # Add AI response to history
    st.session_state.chat_history.append(("AI", response))
    st.rerun()

# Action buttons
if st.session_state.get("chat_started") and st.session_state.chat_history:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col2:
        if st.button("🔄 Reset Chat"):
            st.session_state.chat_history = []
            st.session_state.last_question = None
            st.rerun()
    with col3:
        chat_history_json = json.dumps(st.session_state.chat_history, indent=4)
        st.download_button(
            "📥 Download Chat History",
            chat_history_json,
            "chat_history.json",
            "application/json"
        )