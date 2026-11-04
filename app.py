
import streamlit as st
import requests


API_URL = "https://solstice-docs-api.onrender.com/ask"


st.set_page_config(
    page_title="Solstice Docs Assistant",
    page_icon="📚",
    layout="wide"
)


# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("Solstice Learning")

    st.caption("Documentation Q&A Assistant")

    st.divider()

    st.subheader("Frequently Asked Questions")

    faqs = [
        "How do I reset my password?",
        "How long is the password reset link valid?",
        "How many rows can I export to CSV?",
        "What are the API rate limits for Growth?",
        "How do I invite team members?"
    ]

    for faq in faqs:

        if st.button(faq, use_container_width=True):
            st.session_state.pending_question = faq

    st.divider()

    st.subheader("Documentation Topics")

    topics = [
        "Password Management",
        "Team Management",
        "CSV Report Exports",
        "Billing and Subscriptions",
        "Single Sign-On (SSO)",
        "API Rate Limits"
    ]

    for topic in topics:
        st.markdown(f"- {topic}")

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending_question", None)
        st.rerun()


# -----------------------------
# Main Chat Interface
# -----------------------------

st.title("Docs Q&A Assistant")

st.caption(
    "Ask questions about Solstice Learning. "
    "Answers are generated using the available documentation."
)


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Question Processing
# -----------------------------

question = st.chat_input("Ask a question about Solstice Learning...")

# FAQ button selection
if "pending_question" in st.session_state:

    question = st.session_state.pop("pending_question")


if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Call FastAPI backend
    with st.chat_message("assistant"):

        with st.spinner("Searching documentation..."):

            try:

                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=120
                )

                response.raise_for_status()

                result = response.json()

                answer = result["answer"]

                st.markdown(answer)

                if result["status"] == "abstained":
                    st.caption(
                        "The available documentation does not "
                        "contain sufficient information."
                    )

            except requests.exceptions.RequestException:

                answer = (
                    "Unable to connect to the assistant. "
                    "Please ensure the backend is running and try again."
                )

                st.error(answer)

            except (ValueError, KeyError):

                answer = (
                    "An unexpected response was received. "
                    "Please try again."
                )

                st.error(answer)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
