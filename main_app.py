"""Intelligent Streamlit dashboard for renewable energy projects."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


APP_TITLE = "Renewable Energy Intelligence Hub"
MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "energia_renovable.csv"
PROMPT_FILE = BASE_DIR / "system_prompt.txt"
PRIMARY_COLOR = "#0C4D78"
SECONDARY_COLOR = "#00A896"
ACCENT_COLOR = "#E76F51"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(12, 77, 120, 0.14), transparent 30%),
            radial-gradient(circle at top right, rgba(0, 168, 150, 0.12), transparent 28%),
            linear-gradient(180deg, #f7fafc 0%, #edf4f7 100%);
    }
    .hero-box {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(12, 77, 120, 0.10);
        border-radius: 20px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 14px 34px rgba(12, 77, 120, 0.08);
    }
    .insight-box {
        background: linear-gradient(135deg, rgba(12, 77, 120, 0.08), rgba(0, 168, 150, 0.07));
        border: 1px solid rgba(12, 77, 120, 0.12);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin: 0.5rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(uploaded_file=None) -> pd.DataFrame:
    source = uploaded_file if uploaded_file is not None else DATA_FILE
    return pd.read_csv(source)


@st.cache_data
def load_system_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8").strip()


def initialize_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = data.columns.str.strip()
    data["Fecha_Entrada_Operacion"] = pd.to_datetime(
        data["Fecha_Entrada_Operacion"], errors="coerce"
    )
    data["Conectado_SIN"] = (
        data["Conectado_SIN"].astype(str).str.lower().map({"true": True, "false": False})
    )
    for column in [
        "Capacidad_Instalada_MW",
        "Generacion_Diaria_MWh",
        "Eficiencia_Planta_Pct",
        "Inversion_Inicial_MUSD",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def format_number(value: float, decimals: int = 1) -> str:
    return f"{value:,.{decimals}f}"


def add_insight(message: str) -> None:
    st.markdown(f'<div class="insight-box">{message}</div>', unsafe_allow_html=True)


def build_sidebar(base_df: pd.DataFrame) -> dict:
    with st.sidebar:
        st.title("Dashboard Controls")
        api_key = st.text_input("Groq API Key", type="password")
        uploaded_file = st.file_uploader("Optional CSV override", type=["csv"])
        language = st.selectbox("AI response language", ["Auto", "Spanish", "English"], index=0)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.3, step=0.1)
        max_tokens = st.slider("Max tokens", min_value=256, max_value=2048, value=700, step=128)

        technologies = sorted(base_df["Tecnologia"].dropna().unique().tolist())
        operators = sorted(base_df["Operador"].dropna().unique().tolist())
        states = sorted(base_df["Estado_Actual"].dropna().unique().tolist())

        st.markdown("---")
        st.markdown("### Filters")
        technology_sel = st.multiselect("Technology", technologies, default=technologies)
        operator_sel = st.multiselect("Operator", operators, default=operators)
        state_sel = st.multiselect("Project state", states, default=states)
        connected_sel = st.multiselect("Connected to SIN", [True, False], default=[True, False])

        min_date = base_df["Fecha_Entrada_Operacion"].min()
        max_date = base_df["Fecha_Entrada_Operacion"].max()
        date_range = None
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.date_input(
                "Operation start date range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    return {
        "api_key": api_key,
        "uploaded_file": uploaded_file,
        "language": language,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "technology_sel": technology_sel,
        "operator_sel": operator_sel,
        "state_sel": state_sel,
        "connected_sel": connected_sel,
        "date_range": date_range,
    }


def filter_data(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    data = df.copy()
    data = data[data["Tecnologia"].isin(filters["technology_sel"])]
    data = data[data["Operador"].isin(filters["operator_sel"])]
    data = data[data["Estado_Actual"].isin(filters["state_sel"])]
    data = data[data["Conectado_SIN"].isin(filters["connected_sel"])]
    if filters["date_range"] and len(filters["date_range"]) == 2:
        start_date, end_date = filters["date_range"]
        data = data[data["Fecha_Entrada_Operacion"].dt.date.between(start_date, end_date)]
    return data


def language_instruction(language: str) -> str:
    if language == "Spanish":
        return "Always answer in Spanish."
    if language == "English":
        return "Always answer in English."
    return "Answer in the same language used by the user."


def build_dataset_context(df: pd.DataFrame) -> str:
    project_count = len(df)
    total_capacity = df["Capacidad_Instalada_MW"].sum()
    total_generation = df["Generacion_Diaria_MWh"].sum()
    avg_efficiency = df["Eficiencia_Planta_Pct"].mean()
    total_investment = df["Inversion_Inicial_MUSD"].sum()
    connected_share = df["Conectado_SIN"].mean() * 100 if project_count else 0
    top_tech = df["Tecnologia"].value_counts().head(3).to_dict()
    top_states = df["Estado_Actual"].value_counts().to_dict()
    top_operators = df["Operador"].value_counts().head(5).to_dict()
    numeric_summary = (
        df[[
            "Capacidad_Instalada_MW",
            "Generacion_Diaria_MWh",
            "Eficiencia_Planta_Pct",
            "Inversion_Inicial_MUSD",
        ]]
        .describe()
        .round(2)
        .to_dict()
    )
    top_projects = (
        df.nlargest(5, "Generacion_Diaria_MWh")[[
            "ID_Proyecto",
            "Tecnologia",
            "Operador",
            "Generacion_Diaria_MWh",
            "Capacidad_Instalada_MW",
            "Estado_Actual",
        ]]
        .to_dict(orient="records")
    )

    min_date = df["Fecha_Entrada_Operacion"].min()
    max_date = df["Fecha_Entrada_Operacion"].max()
    date_range_text = "unknown"
    if pd.notna(min_date) and pd.notna(max_date):
        date_range_text = f"{min_date.date()} to {max_date.date()}"

    return "\n".join(
        [
            "Dataset name: energia_renovable.csv",
            f"Filtered project count: {project_count}",
            f"Date range: {date_range_text}",
            f"Total installed capacity MW: {total_capacity:.2f}",
            f"Total daily generation MWh: {total_generation:.2f}",
            f"Average efficiency pct: {avg_efficiency:.2f}",
            f"Total initial investment MUSD: {total_investment:.2f}",
            f"Connected to SIN share pct: {connected_share:.2f}",
            f"Top technologies: {top_tech}",
            f"Project states: {top_states}",
            f"Top operators: {top_operators}",
            f"Numeric summary: {numeric_summary}",
            f"Top generation projects: {top_projects}",
        ]
    )


def build_messages(user_prompt: str, selected_language: str, dataset_context: str) -> list[dict[str, str]]:
    system_prompt = load_system_prompt()
    messages = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                f"{language_instruction(selected_language)}\n\n"
                "Use the following dataset context as the primary source for data-specific answers. "
                "Do not invent values that are not supported by this context.\n\n"
                f"{dataset_context}"
            ),
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


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-box">
            <h1 style="margin-bottom:0.2rem;">Renewable Energy Intelligence Hub</h1>
            <p style="margin:0; color:#43596b; font-size:1.05rem;">
                Explore the renewable energy dataset with filters, operational metrics, investment views,
                and an Llama 3.3 70B copilot that explains the filtered data when you chat with it.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(df: pd.DataFrame) -> None:
    cols = st.columns(5)
    cols[0].metric("Projects", str(len(df)))
    cols[1].metric("Installed capacity", f"{format_number(df['Capacidad_Instalada_MW'].sum())} MW")
    cols[2].metric("Daily generation", f"{format_number(df['Generacion_Diaria_MWh'].sum())} MWh")
    cols[3].metric("Avg. efficiency", f"{format_number(df['Eficiencia_Planta_Pct'].mean())} %")
    cols[4].metric("Initial investment", f"{format_number(df['Inversion_Inicial_MUSD'].sum())} MUSD")


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Portfolio Overview")
    render_kpis(df)

    connected_share = df["Conectado_SIN"].mean() * 100 if len(df) else 0
    lead_technology = df["Tecnologia"].value_counts().idxmax()
    lead_state = df["Estado_Actual"].value_counts().idxmax()
    add_insight(
        f"The filtered portfolio contains <b>{len(df)}</b> projects. The dominant technology is <b>{lead_technology}</b>, "
        f"the most common state is <b>{lead_state}</b>, and <b>{connected_share:.1f}%</b> of projects are connected to SIN."
    )

    left, right = st.columns(2)
    tech_counts = df["Tecnologia"].value_counts().reset_index()
    tech_counts.columns = ["Technology", "Projects"]
    state_counts = df["Estado_Actual"].value_counts().reset_index()
    state_counts.columns = ["State", "Projects"]

    with left:
        fig = px.bar(
            tech_counts,
            x="Projects",
            y="Technology",
            orientation="h",
            title="Project mix by technology",
            text="Projects",
            color="Projects",
            color_continuous_scale="Tealgrn",
        )
        fig.update_layout(showlegend=False, xaxis_title="Projects", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.pie(
            state_counts,
            values="Projects",
            names="State",
            hole=0.45,
            title="Project state share",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)


def render_operations(df: pd.DataFrame) -> None:
    st.subheader("Operations and Performance")
    left, right = st.columns(2)

    with left:
        monthly = (
            df.assign(Month=df["Fecha_Entrada_Operacion"].dt.to_period("M").dt.to_timestamp())
            .groupby("Month", as_index=False)
            .agg(
                Projects=("ID_Proyecto", "count"),
                Capacity_MW=("Capacidad_Instalada_MW", "sum"),
                Generation_MWh=("Generacion_Diaria_MWh", "sum"),
            )
        )
        fig = px.line(
            monthly,
            x="Month",
            y=["Projects", "Capacity_MW", "Generation_MWh"],
            markers=True,
            title="Monthly evolution of projects, capacity and generation",
            color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR],
        )
        fig.update_layout(yaxis_title="Monthly total", legend_title_text="Metric")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            df,
            x="Capacidad_Instalada_MW",
            y="Generacion_Diaria_MWh",
            color="Tecnologia",
            size="Inversion_Inicial_MUSD",
            hover_data=["ID_Proyecto", "Operador", "Estado_Actual"],
            title="Capacity vs daily generation",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            xaxis_title="Installed capacity (MW)",
            yaxis_title="Daily generation (MWh)",
            legend_title_text="Technology",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_investment(df: pd.DataFrame) -> None:
    st.subheader("Investment and Efficiency")
    left, right = st.columns(2)

    with left:
        investment_by_operator = (
            df.groupby("Operador", as_index=False)["Inversion_Inicial_MUSD"]
            .sum()
            .sort_values("Inversion_Inicial_MUSD", ascending=False)
        )
        fig = px.bar(
            investment_by_operator,
            x="Operador",
            y="Inversion_Inicial_MUSD",
            title="Initial investment by operator",
            color="Inversion_Inicial_MUSD",
            color_continuous_scale="Blues",
        )
        fig.update_layout(xaxis_title="Operator", yaxis_title="Investment (MUSD)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        efficiency_by_tech = (
            df.groupby("Tecnologia", as_index=False)["Eficiencia_Planta_Pct"]
            .mean()
            .sort_values("Eficiencia_Planta_Pct", ascending=False)
        )
        fig = px.bar(
            efficiency_by_tech,
            x="Eficiencia_Planta_Pct",
            y="Tecnologia",
            orientation="h",
            title="Average efficiency by technology",
            text="Eficiencia_Planta_Pct",
            color="Eficiencia_Planta_Pct",
            color_continuous_scale="Emrld",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(xaxis_title="Efficiency (%)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)


def render_data_table(df: pd.DataFrame) -> None:
    st.subheader("Filtered dataset")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download filtered CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="renewable_energy_filtered.csv",
        mime="text/csv",
    )


def render_ai_summary(settings: dict, dataset_context: str) -> None:
    st.subheader("AI executive interpretation")
    st.caption("Generate a concise interpretation of the currently filtered data using Llama 3.3 70B.")

    if not settings["api_key"]:
        st.info("Enter your Groq API key in the sidebar to generate AI interpretations.")
        return

    if st.button("Generate AI interpretation"):
        prompt = (
            "Explain the filtered renewable energy dataset as an executive summary. "
            "Highlight the portfolio size, dominant technologies, operational readiness, investment profile, "
            "performance signals, and one or two risks or caveats from the data."
        )
        with st.spinner("Generating interpretation..."):
            try:
                summary = request_groq_completion(
                    api_key=settings["api_key"],
                    messages=build_messages(prompt, settings["language"], dataset_context),
                    temperature=settings["temperature"],
                    max_tokens=settings["max_tokens"],
                )
                add_insight(summary)
            except requests.HTTPError as exc:
                st.error(f"Groq API error: {exc.response.status_code} - {exc.response.text}")
            except requests.RequestException as exc:
                st.error(f"Request error: {exc}")
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                st.error(f"Unexpected response format: {exc}")


def render_chat(settings: dict, dataset_context: str) -> None:
    st.subheader("Ask the dataset")
    st.caption("Chat with Llama 3.3 70B about the filtered data currently shown in the dashboard.")

    if not st.session_state.messages:
        add_insight(
            "Ask questions like: Which technology dominates this filtered portfolio? Which operators invest the most? "
            "What does the relationship between capacity and generation suggest?"
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask something about the filtered renewable energy data...")
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
        with st.spinner("Analyzing filtered data..."):
            try:
                assistant_reply = request_groq_completion(
                    api_key=settings["api_key"],
                    messages=build_messages(user_prompt, settings["language"], dataset_context),
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


def main() -> None:
    initialize_session()
    render_header()

    base_df = clean_data(load_data())
    settings = build_sidebar(base_df)
    dataset = clean_data(load_data(settings["uploaded_file"]))
    filtered_df = filter_data(dataset, settings)

    if filtered_df.empty:
        st.warning("No records match the current filters.")
        return

    dataset_context = build_dataset_context(filtered_df)

    tab_overview, tab_operations, tab_investment, tab_ai_summary, tab_chat, tab_data = st.tabs(
        ["Overview", "Operations", "Investment", "AI Summary", "AI Chat", "Data"]
    )

    with tab_overview:
        render_overview(filtered_df)

    with tab_operations:
        render_operations(filtered_df)

    with tab_investment:
        render_investment(filtered_df)

    with tab_ai_summary:
        render_ai_summary(settings, dataset_context)

    with tab_chat:
        render_chat(settings, dataset_context)

    with tab_data:
        render_data_table(filtered_df)


if __name__ == "__main__":
    main()