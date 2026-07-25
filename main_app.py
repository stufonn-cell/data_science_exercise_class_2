"""Streamlit app that transforms interviews into tabular data and analyzes them."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


APP_TITLE = "Interview Intelligence Studio"
MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_PROMPT_FILE = BASE_DIR / "system_prompt.txt"
EXTRACTION_PROMPT_FILE = BASE_DIR / "extraction_prompt.txt"
PRIMARY_COLOR = "#0B4F6C"
SECONDARY_COLOR = "#01BAEF"
ACCENT_COLOR = "#20BF55"

EXPECTED_COLUMNS = [
    "interview_id",
    "respondent_id",
    "respondent_profile",
    "location",
    "topic",
    "pain_point",
    "current_solution",
    "desired_outcome",
    "sentiment",
    "urgency_score",
    "budget_score",
    "adoption_readiness_score",
    "confidence_score",
    "quote",
]

NUMERIC_COLUMNS = [
    "urgency_score",
    "budget_score",
    "adoption_readiness_score",
    "confidence_score",
]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(11, 79, 108, 0.12), transparent 28%),
            radial-gradient(circle at top right, rgba(1, 186, 239, 0.10), transparent 26%),
            linear-gradient(180deg, #f6fbfd 0%, #eef5f8 100%);
    }
    .hero-box {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(11, 79, 108, 0.10);
        border-radius: 20px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 32px rgba(11, 79, 108, 0.08);
    }
    .insight-box {
        background: linear-gradient(135deg, rgba(11, 79, 108, 0.07), rgba(32, 191, 85, 0.06));
        border: 1px solid rgba(11, 79, 108, 0.12);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin: 0.5rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_prompt(prompt_path: str) -> str:
    return Path(prompt_path).read_text(encoding="utf-8").strip()


def initialize_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "structured_records" not in st.session_state:
        st.session_state.structured_records = []
    if "analysis_summary" not in st.session_state:
        st.session_state.analysis_summary = ""


def add_insight(message: str) -> None:
    st.markdown(f'<div class="insight-box">{message}</div>', unsafe_allow_html=True)


def format_number(value: float, decimals: int = 1) -> str:
    return f"{value:,.{decimals}f}"


def build_sidebar(has_data: bool, df: pd.DataFrame | None = None) -> dict:
    with st.sidebar:
        st.title("AI Controls")
        api_key = st.text_input("Groq API Key", type="password")
        language = st.selectbox("Response language", ["Auto", "Spanish", "English"], index=0)
        extraction_temperature = st.slider(
            "Extraction temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
        )
        analysis_temperature = st.slider(
            "Analysis temperature",
            min_value=0.0,
            max_value=1.5,
            value=0.3,
            step=0.1,
        )
        max_tokens = st.slider("Max tokens", min_value=256, max_value=2048, value=900, step=128)
        uploaded_file = st.file_uploader("Optional interview text file", type=["txt", "md"])

        filters = {
            "topics": [],
            "sentiments": [],
            "profiles": [],
            "locations": [],
        }

        if has_data and df is not None and not df.empty:
            st.markdown("---")
            st.markdown("### EDA Filters")
            filters["topics"] = st.multiselect(
                "Topic",
                sorted(df["topic"].dropna().astype(str).unique().tolist()),
                default=sorted(df["topic"].dropna().astype(str).unique().tolist()),
            )
            filters["sentiments"] = st.multiselect(
                "Sentiment",
                sorted(df["sentiment"].dropna().astype(str).unique().tolist()),
                default=sorted(df["sentiment"].dropna().astype(str).unique().tolist()),
            )
            filters["profiles"] = st.multiselect(
                "Respondent profile",
                sorted(df["respondent_profile"].dropna().astype(str).unique().tolist()),
                default=sorted(df["respondent_profile"].dropna().astype(str).unique().tolist()),
            )
            filters["locations"] = st.multiselect(
                "Location",
                sorted(df["location"].dropna().astype(str).unique().tolist()),
                default=sorted(df["location"].dropna().astype(str).unique().tolist()),
            )

        if st.button("Clear AI chat"):
            st.session_state.messages = []
            st.rerun()

    return {
        "api_key": api_key,
        "language": language,
        "extraction_temperature": extraction_temperature,
        "analysis_temperature": analysis_temperature,
        "max_tokens": max_tokens,
        "uploaded_file": uploaded_file,
        "filters": filters,
    }


def language_instruction(language: str) -> str:
    if language == "Spanish":
        return "Always answer in Spanish."
    if language == "English":
        return "Always answer in English."
    return "Answer in the same language used by the user."


def decode_uploaded_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


def request_groq_completion(
    api_key: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    response = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def parse_json_payload(raw_text: str) -> list[dict]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

    opening_positions = [index for index in [cleaned.find("{"), cleaned.find("[")] if index != -1]
    first_brace = min(opening_positions, default=-1)
    last_brace = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if first_brace != -1 and last_brace != -1:
        cleaned = cleaned[first_brace : last_brace + 1]

    payload = json.loads(cleaned)
    if isinstance(payload, dict):
        records = payload.get("records", [])
    elif isinstance(payload, list):
        records = payload
    else:
        records = []

    if not isinstance(records, list):
        raise ValueError("The model response does not contain a valid records list.")
    return records


def normalize_records(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {column: record.get(column) for column in EXPECTED_COLUMNS}
        rows.append(row)

    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    if df.empty:
        return df

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    text_columns = [
        "interview_id",
        "respondent_id",
        "respondent_profile",
        "location",
        "topic",
        "pain_point",
        "current_solution",
        "desired_outcome",
        "sentiment",
        "quote",
    ]
    for column in text_columns:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()

    return df


def build_extraction_messages(transcript_text: str, selected_language: str) -> list[dict[str, str]]:
    extraction_prompt = load_prompt(str(EXTRACTION_PROMPT_FILE))
    return [
        {
            "role": "system",
            "content": f"{extraction_prompt}\n\n{language_instruction(selected_language)}",
        },
        {"role": "user", "content": transcript_text},
    ]


def build_dataset_context(df: pd.DataFrame) -> str:
    topic_counts = df["topic"].value_counts().head(8).to_dict()
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    profile_counts = df["respondent_profile"].value_counts().head(8).to_dict()
    location_counts = df["location"].value_counts().head(8).to_dict()
    pain_points = df["pain_point"].value_counts().head(10).to_dict()
    desired_outcomes = df["desired_outcome"].value_counts().head(10).to_dict()
    numeric_summary = df[NUMERIC_COLUMNS].describe().round(2).to_dict()
    sample_records = df.head(12).to_dict(orient="records")

    return "\n".join(
        [
            "Dataset origin: AI-structured interview transcript",
            f"Structured rows: {len(df)}",
            f"Unique interviews: {df['interview_id'].nunique()}",
            f"Unique respondents: {df['respondent_id'].nunique()}",
            f"Topics: {topic_counts}",
            f"Sentiments: {sentiment_counts}",
            f"Profiles: {profile_counts}",
            f"Locations: {location_counts}",
            f"Pain points: {pain_points}",
            f"Desired outcomes: {desired_outcomes}",
            f"Numeric summary: {numeric_summary}",
            f"Sample structured records: {sample_records}",
        ]
    )


def build_analysis_messages(user_prompt: str, selected_language: str, dataset_context: str) -> list[dict[str, str]]:
    system_prompt = load_prompt(str(ANALYSIS_PROMPT_FILE))
    messages = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                f"{language_instruction(selected_language)}\n\n"
                "Use the following structured interview dataset context as the primary source for your answer. "
                "Do not invent counts or scores beyond the supplied context.\n\n"
                f"{dataset_context}"
            ),
        }
    ]
    messages.extend(st.session_state.messages)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def filter_structured_data(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    filtered = df.copy()
    if filters["topics"]:
        filtered = filtered[filtered["topic"].isin(filters["topics"])]
    if filters["sentiments"]:
        filtered = filtered[filtered["sentiment"].isin(filters["sentiments"])]
    if filters["profiles"]:
        filtered = filtered[filtered["respondent_profile"].isin(filters["profiles"])]
    if filters["locations"]:
        filtered = filtered[filtered["location"].isin(filters["locations"])]
    return filtered


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-box">
            <h1 style="margin-bottom:0.2rem;">Interview Intelligence Studio</h1>
            <p style="margin:0; color:#43596b; font-size:1.05rem;">
                Paste an interview transcript, transform it into structured tabular data with Llama 3.3 70B,
                and run a strong Streamlit EDA with AI explanations on top of the extracted dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ingestion(settings: dict) -> pd.DataFrame | None:
    st.subheader("1. Interview to table")
    st.caption("Paste one or more interviews. The model will convert the text into structured rows for analysis.")

    uploaded_text = decode_uploaded_text(settings["uploaded_file"])
    transcript_text = st.text_area(
        "Interview transcript",
        value=uploaded_text,
        height=260,
        placeholder="Paste the interview transcript here. The AI will extract one structured row per relevant insight.",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        transform_clicked = st.button("Transform into table", type="primary")
    with col2:
        if transcript_text.strip():
            st.caption(f"Transcript length: {len(transcript_text.split())} words")

    if not transform_clicked:
        if st.session_state.structured_records:
            return normalize_records(st.session_state.structured_records)
        return None

    if not settings["api_key"]:
        st.warning("Enter your Groq API key in the sidebar before transforming the interview.")
        return None

    if not transcript_text.strip():
        st.warning("Paste interview text before running the transformation.")
        return None

    with st.spinner("Transforming transcript into structured rows..."):
        try:
            raw_response = request_groq_completion(
                api_key=settings["api_key"],
                messages=build_extraction_messages(transcript_text, settings["language"]),
                temperature=settings["extraction_temperature"],
                max_tokens=settings["max_tokens"],
            )
            records = parse_json_payload(raw_response)
            structured_df = normalize_records(records)
            if structured_df.empty:
                st.error("The model returned no structured rows. Try a clearer transcript or a smaller chunk of text.")
                return None
            st.session_state.structured_records = structured_df.to_dict(orient="records")
            st.session_state.messages = []
            st.session_state.analysis_summary = ""
            st.success(f"Structured dataset created with {len(structured_df)} rows.")
            return structured_df
        except requests.HTTPError as exc:
            st.error(f"Groq API error: {exc.response.status_code} - {exc.response.text}")
        except requests.RequestException as exc:
            st.error(f"Request error: {exc}")
        except (json.JSONDecodeError, ValueError) as exc:
            st.error(f"The AI response could not be parsed into structured JSON: {exc}")

    return None


def render_kpis(df: pd.DataFrame) -> None:
    cols = st.columns(5)
    cols[0].metric("Structured rows", str(len(df)))
    cols[1].metric("Interviews", str(df["interview_id"].nunique()))
    cols[2].metric("Respondents", str(df["respondent_id"].nunique()))
    cols[3].metric("Avg. urgency", format_number(df["urgency_score"].mean()))
    cols[4].metric("Avg. adoption", format_number(df["adoption_readiness_score"].mean()))


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("2. EDA Overview")
    render_kpis(df)

    top_topic = df["topic"].value_counts().idxmax()
    top_sentiment = df["sentiment"].value_counts().idxmax()
    add_insight(
        f"The dataset currently contains <b>{len(df)}</b> structured insights. The most frequent topic is <b>{top_topic}</b> and the dominant sentiment is <b>{top_sentiment}</b>."
    )

    left, right = st.columns(2)
    topic_counts = df["topic"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    with left:
        fig = px.bar(
            topic_counts,
            x="Count",
            y="Topic",
            orientation="h",
            title="Topic frequency",
            text="Count",
            color="Count",
            color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, xaxis_title="Insights", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.pie(
            sentiment_counts,
            values="Count",
            names="Sentiment",
            hole=0.45,
            title="Sentiment mix",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)


def render_needs_and_profiles(df: pd.DataFrame) -> None:
    st.subheader("3. Needs, pains, and profiles")
    left, right = st.columns(2)

    with left:
        pain_counts = df["pain_point"].value_counts().head(10).reset_index()
        pain_counts.columns = ["Pain point", "Count"]
        fig = px.bar(
            pain_counts,
            x="Count",
            y="Pain point",
            orientation="h",
            title="Top pain points",
            text="Count",
            color="Count",
            color_continuous_scale="Reds",
        )
        fig.update_layout(showlegend=False, xaxis_title="Mentions", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        profile_counts = df["respondent_profile"].value_counts().head(10).reset_index()
        profile_counts.columns = ["Profile", "Count"]
        fig = px.bar(
            profile_counts,
            x="Profile",
            y="Count",
            title="Respondent profile distribution",
            color="Count",
            color_continuous_scale="Tealgrn",
        )
        fig.update_layout(xaxis_title="Profile", yaxis_title="Insights")
        st.plotly_chart(fig, use_container_width=True)


def render_scores(df: pd.DataFrame) -> None:
    st.subheader("4. Scoring analysis")
    left, right = st.columns(2)

    with left:
        topic_scores = (
            df.groupby("topic", as_index=False)[["urgency_score", "adoption_readiness_score"]]
            .mean()
            .sort_values("urgency_score", ascending=False)
        )
        fig = px.bar(
            topic_scores,
            x="topic",
            y=["urgency_score", "adoption_readiness_score"],
            barmode="group",
            title="Average urgency vs adoption readiness by topic",
            color_discrete_sequence=[PRIMARY_COLOR, ACCENT_COLOR],
        )
        fig.update_layout(xaxis_title="Topic", yaxis_title="Average score")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            df,
            x="budget_score",
            y="urgency_score",
            color="sentiment",
            size="confidence_score",
            hover_data=["topic", "pain_point", "desired_outcome"],
            title="Budget vs urgency",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig.update_layout(xaxis_title="Budget score", yaxis_title="Urgency score")
        st.plotly_chart(fig, use_container_width=True)


def render_ai_summary(settings: dict, dataset_context: str) -> None:
    st.subheader("5. AI summary")
    st.caption("Generate an executive interpretation of the transformed interview dataset.")

    if not settings["api_key"]:
        st.info("Enter your Groq API key in the sidebar to generate AI interpretations.")
        return

    if st.button("Generate AI summary"):
        prompt = (
            "Summarize the structured interview dataset. Explain the dominant themes, pain points, sentiment patterns, "
            "respondent segments, urgency profile, willingness-to-pay signals, and what actions a team should take next."
        )
        with st.spinner("Generating AI summary..."):
            try:
                summary = request_groq_completion(
                    api_key=settings["api_key"],
                    messages=build_analysis_messages(prompt, settings["language"], dataset_context),
                    temperature=settings["analysis_temperature"],
                    max_tokens=settings["max_tokens"],
                )
                st.session_state.analysis_summary = summary
            except requests.HTTPError as exc:
                st.error(f"Groq API error: {exc.response.status_code} - {exc.response.text}")
            except requests.RequestException as exc:
                st.error(f"Request error: {exc}")
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                st.error(f"Unexpected response format: {exc}")

    if st.session_state.analysis_summary:
        add_insight(st.session_state.analysis_summary)


def render_ai_chat(settings: dict, dataset_context: str) -> None:
    st.subheader("6. Ask the dataset")
    st.caption("Chat with Llama 3.3 70B about the tabular dataset created from the interview text.")

    if not st.session_state.messages:
        add_insight(
            "Ask questions like: Which themes appear most often? Which pain points have the highest urgency? Which respondent profiles seem most ready to adopt a solution?"
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask something about the structured interview data...")
    if not user_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    if not settings["api_key"]:
        warning_message = "Enter your Groq API key in the sidebar before asking the AI to interpret the data."
        st.session_state.messages.append({"role": "assistant", "content": warning_message})
        with st.chat_message("assistant"):
            st.warning(warning_message)
        return

    with st.chat_message("assistant"):
        with st.spinner("Interpreting structured interview data..."):
            try:
                assistant_reply = request_groq_completion(
                    api_key=settings["api_key"],
                    messages=build_analysis_messages(user_prompt, settings["language"], dataset_context),
                    temperature=settings["analysis_temperature"],
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


def render_data_table(df: pd.DataFrame) -> None:
    st.subheader("7. Structured table")
    edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
    st.download_button(
        "Download structured CSV",
        data=edited_df.to_csv(index=False).encode("utf-8"),
        file_name="structured_interview_data.csv",
        mime="text/csv",
    )


def main() -> None:
    initialize_session()
    render_header()

    existing_df = normalize_records(st.session_state.structured_records) if st.session_state.structured_records else None
    settings = build_sidebar(existing_df is not None and not existing_df.empty, existing_df)

    structured_df = render_ingestion(settings)
    if structured_df is None or structured_df.empty:
        return

    filtered_df = filter_structured_data(structured_df, settings["filters"])
    if filtered_df.empty:
        st.warning("No structured rows match the current filters.")
        return

    dataset_context = build_dataset_context(filtered_df)

    tab_overview, tab_needs, tab_scores, tab_summary, tab_chat, tab_data = st.tabs(
        ["Overview", "Needs", "Scores", "AI Summary", "AI Chat", "Data"]
    )

    with tab_overview:
        render_overview(filtered_df)

    with tab_needs:
        render_needs_and_profiles(filtered_df)

    with tab_scores:
        render_scores(filtered_df)

    with tab_summary:
        render_ai_summary(settings, dataset_context)

    with tab_chat:
        render_ai_chat(settings, dataset_context)

    with tab_data:
        render_data_table(filtered_df)


if __name__ == "__main__":
    main()