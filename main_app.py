"""Streamlit chatbot powered by Groq and Llama 3.3 70B."""

from __future__ import annotations

from pathlib import Path

import requests
import streamlit as st


APP_TITLE = "World History Chatbot"
MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
BASE_DIR = Path(__file__).resolve().parent
PROMPT_FILE = BASE_DIR / "system_prompt.txt"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(5, 117, 230, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(0, 168, 150, 0.10), transparent 26%),
            linear-gradient(180deg, #f4f8fb 0%, #eef4f6 100%);
    }
    .hero-box {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(12, 77, 120, 0.10);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(31, 78, 121, 0.07);
    }
    .info-box {
        background: rgba(12, 77, 120, 0.06);
        border-left: 4px solid #0c4d78;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 0.75rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_system_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


def initialize_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def build_sidebar() -> dict:
    with st.sidebar:
        st.title("Configuration")
        api_key = st.text_input("Groq API Key", type="password")

        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.4, step=0.1)
        max_tokens = st.slider("Max tokens", min_value=128, max_value=2048, value=512, step=128)
        language = st.selectbox("Response language", ["Auto", "Spanish", "English"], index=0)

        st.markdown("---")
        st.markdown("### Scope")
        st.caption("World history, civilizations, wars, revolutions, geography, literature, art, science, and general culture.")

        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    return {
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "language": language,
    }


def language_instruction(language: str) -> str:
    if language == "Spanish":
        return "Always answer in Spanish."
    if language == "English":
        return "Always answer in English."
    return "Answer in the same language used by the user."


def build_messages(user_prompt: str, selected_language: str) -> list[dict[str, str]]:
    system_prompt = load_system_prompt()
    messages = [
        {
            "role": "system",
            "content": f"{system_prompt}\n\n{language_instruction(selected_language)}",
        }
    ]

    messages.extend(st.session_state.messages)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def request_groq_completion(
    api_key: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        GROQ_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-box">
            <h1 style="margin-bottom:0.25rem;">World History and General Culture Chatbot</h1>
            <p style="margin:0; font-size:1.05rem; color:#415466;">
                Ask about world history, empires, revolutions, geography, literature, art, science,
                or broad general culture using Groq with Llama 3.3 70B.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-box">
            Enter your Groq API key directly in the dashboard sidebar before starting the conversation.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_examples() -> None:
    with st.expander("Example prompts"):
        st.markdown(
            """
            - What were the main causes of World War I?
            - Explain the difference between the Renaissance and the Enlightenment.
            - Give me a short timeline of Ancient Egypt.
            - Who was Simón Bolívar and why is he important?
            - What was the cultural impact of the Silk Road?
            """
        )


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_empty_state() -> None:
    if st.session_state.messages:
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Suggested topics")
        st.markdown(
            """
            - Ancient civilizations
            - Middle Ages and early modern Europe
            - Latin American independence
            - World wars and Cold War
            - Art, religion, and scientific change
            """
        )
    with col2:
        st.markdown("### Good question style")
        st.markdown(
            """
            - Ask for a comparison
            - Ask for a timeline
            - Ask for causes and consequences
            - Ask for a short or detailed answer
            - Ask in Spanish or English
            """
        )


def main() -> None:
    initialize_session()
    render_header()
    render_examples()
    settings = build_sidebar()
    render_empty_state()
    render_chat_history()

    user_prompt = st.chat_input("Ask a question about world history or general culture...")
    if not user_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    if not settings["api_key"]:
        warning_message = "Enter your Groq API key in the sidebar before sending a question."
        st.session_state.messages.append({"role": "assistant", "content": warning_message})
        with st.chat_message("assistant"):
            st.warning(warning_message)
        return

    messages = build_messages(user_prompt, settings["language"])

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                assistant_reply = request_groq_completion(
                    api_key=settings["api_key"],
                    messages=messages,
                    temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"],
                )
                st.markdown(assistant_reply)
            except requests.HTTPError as exc:
                assistant_reply = f"Groq API error: {exc.response.status_code} - {exc.response.text}"
                st.error(assistant_reply)
            except requests.RequestException as exc:
                assistant_reply = f"Request error: {exc}"
                st.error(assistant_reply)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                assistant_reply = f"Unexpected response format: {exc}"
                st.error(assistant_reply)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    main()