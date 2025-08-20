import streamlit as st
import os
import sys

# Add src/ to Python path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Import our custom modules
try:
    from chatbot import FAQChatbot
except ImportError as e:
    st.error(f"Import error: {e}. Make sure src/ files exist and names are correct.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🍽️ Restaurant FAQ Chatbot",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS (improved readability)
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #ff6b6b, #feca57);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #000000;  /* force black text for readability */
    }
    .user-message { 
        background-color: #d1ecf1;  /* light blue */
        border-left: 4px solid #0c5460;
    }
    .bot-message { 
        background-color: #f8d7da;  /* light pink/red */
        border-left: 4px solid #721c24;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ff9800; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Chatbot Initialization ----------
def initialize_chatbot():
    """Initialize chatbot using faqs.json from data/"""
    faq_file = os.path.join(
        os.path.dirname(__file__), "..", "Data", "restaurant_faqs.json"
    )

    if not os.path.exists(faq_file):
        st.error("❌ FAQ file not found in Data/restaurant_faqs.json")
        st.stop()

    try:
        chatbot = FAQChatbot(faq_file)
        return chatbot
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None


# ---------- Example Questions ----------
def display_example_questions():
    """Sidebar with static predefined example questions."""
    st.sidebar.header("💡 Example Questions")
    st.sidebar.markdown("Click on an example to try it:")

    example_questions = [
        "What are your restaurant hours?",
        "Do you offer delivery service?",
        "Do you have vegetarian options?",
        "Do I need a reservation?",
        "Do you have gluten-free options?",
        "Can I order takeout?",
    ]

    for q in example_questions:
        if st.sidebar.button(q):
            st.session_state.user_input = q


# ---------- Main UI ----------
def main():
    st.markdown(
        '<div class="main-header"><h1>🍽️ Restaurant FAQ Chatbot</h1></div>',
        unsafe_allow_html=True,
    )

    chatbot = initialize_chatbot()
    if chatbot is None:
        return

    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Sidebar example questions
    display_example_questions()

    # Input box
    user_input = st.text_input(
        "💬 Ask me anything about the restaurant:",
        value=st.session_state.user_input,
        key="input_box",
    )

    if user_input:
        result = chatbot.get_response(user_input)

        # Confidence levels
        confidence = result["confidence"]
        if confidence >= 0.6:
            conf_label = f"<span class='confidence-high'>High ({confidence:.2f})</span>"
        elif confidence >= 0.3:
            conf_label = (
                f"<span class='confidence-medium'>Medium ({confidence:.2f})</span>"
            )
        else:
            conf_label = f"<span class='confidence-low'>Low ({confidence:.2f})</span>"

        answer = result["answer"] + f"<br><small>Confidence: {conf_label}</small>"

        # Save chat
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", answer))

        st.session_state.user_input = ""  # reset

    # Display chat history
    for role, msg in st.session_state.chat_history:
        css_class = "user-message" if role == "user" else "bot-message"
        st.markdown(
            f'<div class="chat-message {css_class}">{msg}</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
