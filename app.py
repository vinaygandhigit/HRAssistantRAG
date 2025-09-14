import streamlit as st
import requests

# FastAPI endpoint URL (adjust if deployed elsewhere)
API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="HR Policy Assistant", page_icon="🤖", layout="wide")

# Custom CSS for white background and Grok-like styling
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;  /* White background */
        color: #000000;  /* Black text for readability */
        font-size: 18px;  /* Base font size for app content */
        font-family: Arial, sans-serif;
    }
    .stTextInput > div > div > input {
        background-color: #f0f0f0;  /* Light gray input field */
        color: #000000;
        border: 1px solid #cccccc;  /* Light border */
        font-size: 18px;  /* Larger input text */
        padding: 10px;  /* More padding for comfort */
    }
    .stButton > button {
        background-color: #0066cc;  /* Blue button for contrast */
        color: #ffffff;
        border: none;
        border-radius: 5px;
        font-size: 16px;  /* Larger button text */
        padding: 8px 16px;
    }
    .stButton > button:disabled {
        background-color: #cccccc;  /* Gray when disabled */
        color: #666666;
    }
    .user-message {
        background-color: #0066cc;  /* Blue for user messages */
        color: #ffffff;
        padding: 12px;
        border-radius: 10px;
        margin: 5px;
        max-width: 80%;
        align-self: flex-end;
        font-size: 18px;  /* Larger chat text */
    }
    .bot-message {
        background-color: #e0e0e0;  /* Light gray for bot messages */
        color: #000000;
        padding: 12px;
        border-radius: 10px;
        margin: 5px;
        max-width: 80%;
        align-self: flex-start;
        font-size: 18px;  /* Larger chat text */
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    .stSpinner > div {
        font-size: 16px;  /* Larger spinner text */
    }
    h1 {
        font-size: 32px !important;  /* Larger title */
    }
    .stMarkdown {
        font-size: 18px;  /* Larger markdown text */
    }
    </style>
""", unsafe_allow_html=True)

# Session state for chat history and processing flag
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

st.title("HR Policy Assistant 🤖")
st.markdown("Ask questions about HR policies. Powered by RAG with fine-tuned GPT-2.")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)

# Input form at the bottom
with st.form(key="query_form", clear_on_submit=True):
    query = st.text_input("Your question:", key="user_input")
    submit_button = st.form_submit_button(label="Send", disabled=st.session_state.is_processing)

if submit_button and query and not st.session_state.is_processing:
    # Set processing flag
    st.session_state.is_processing = True
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Show loading spinner
    with st.spinner("Processing your query..."):
        try:
            response = requests.post(
                API_URL,
                json={
                    "query": query,
                    "k": 5,
                    "max_new_tokens": 150,
                    "temperature": 0.7
                },
                timeout=60  # Timeout after 30 seconds
            )
            response.raise_for_status()
            bot_response = response.json()["response"]
        except Exception as e:
            bot_response = f"Error: {str(e)}"
    
    # Add bot response to history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # Reset processing flag
    st.session_state.is_processing = False
    
    # Rerun to update chat
    st.rerun()

# Clear chat button
if st.button("Clear Chat", disabled=st.session_state.is_processing):
    st.session_state.messages = []
    st.session_state.is_processing = False
    st.rerun()