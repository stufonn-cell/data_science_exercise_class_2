"""Streamlit EDA application for renewable energy projects."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import streamlit as st


APP_TITLE = "Renewable Energy EDA"
DATA_FILE = Path(__file__).with_name("energia_renovable.csv")
PRIMARY_COLOR = "#1F4E79"
SECONDARY_COLOR = "#2A9D8F"
ACCENT_COLOR = "#E76F51"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", context="talk")

st.markdown(
    """
    <style>
    .stMetric label {
        font-size: 0.85rem !important;
        color: #5f6b7a !important;
    }
    .insight-box {
        background: linear-gradient(135deg, rgba(31,78,121,0.07), rgba(42,157,143,0.07));
        border: 1px solid rgba(31,78,121,0.12);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin: 0.5rem 0 1rem 0;
    }
    .section-title {
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(uploaded_file=None) -> pd.DataFrame:
    """Load the dataset from disk or from an uploaded CSV."""
    source = uploaded_file if uploaded_file is not None else DATA_FILE
    return pd.read_csv(source)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and data types."""
    data = df.copy()
    data.columns = data.columns.str.strip()
    data["Fecha_Entrada_Operacion"] = pd.to_datetime(
        data["Fecha_Entrada_Operacion"], errors="coerce"
    )
    data["Conectado_SIN"] = (
        data["Conectado_SIN"].astype(str).str.lower().map({"true": True, "false": False})
    )
    for column in ["Capacidad_Instalada_MW", "Generacion_Diaria_MWh", "Eficiencia_Planta_Pct", "Inversion_Inicial_MUSD"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def format_number(value: float, decimals: int = 1) -> str:
    return f"{value:,.{decimals}f}"


def add_insight(message: str) -> None:
    st.markdown(f'<div class="insight-box">{message}</div>', unsafe_allow_html=True)


def display_section_header(title: str, subtitle: str) -> None:
    st.markdown(f'<h3 class="section-title">{title}</h3>', unsafe_allow_html=True)
    st.caption(subtitle)


def top_share_text(count: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{(count / total) * 100:.1f}%"


def build_sidebar(base_df: pd.DataFrame) -> dict:
    """Create the filter sidebar and return the selected values."""
    with st.sidebar:
        st.header("Filters")
        uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])

        technologies = sorted(base_df["Tecnologia"].dropna().unique().tolist())
        operators = sorted(base_df["Operador"].dropna().unique().tolist())
        states = sorted(base_df["Estado_Actual"].dropna().unique().tolist())

        technology_sel = st.multiselect("Technology", technologies, default=technologies)
        operator_sel = st.multiselect("Operator", operators, default=operators)
        state_sel = st.multiselect("Project state", states, default=states)
        connected_sel = st.multiselect(
            "Connected to SIN", [True, False], default=[True, False]
        )

        min_date = base_df["Fecha_Entrada_Operacion"].min()
        max_date = base_df["Fecha_Entrada_Operacion"].max()
        date_range = None
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.date_input(
                "Operation entry date range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )

    return {
        "uploaded_file": uploaded_file,
        "technology_sel": technology_sel,
        "operator_sel": operator_sel,
        "state_sel": state_sel,
        "connected_sel": connected_sel,
        "date_range": date_range,
    }


def filter_data(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply all sidebar filters."""
    data = df.copy()
    data = data[data["Tecnologia"].isin(filters["technology_sel"])]
    data = data[data["Operador"].isin(filters["operator_sel"])]
    data = data[data["Estado_Actual"].isin(filters["state_sel"])]
    data = data[data["Conectado_SIN"].isin(filters["connected_sel"])]

    if filters["date_range"] and len(filters["date_range"]) == 2:
        start_date, end_date = filters["date_range"]
        data = data[data["Fecha_Entrada_Operacion"].dt.date.between(start_date, end_date)]

    return data


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    report = df.isna().sum().reset_index()
    report.columns = ["column", "missing_values"]
    return report[report["missing_values"] > 0].sort_values(
        by="missing_values", ascending=False
    )


def duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def render_kpis(df: pd.DataFrame) -> None:
    """Render the top-level KPIs."""
    cols = st.columns(5)
    cols[0].metric("Projects", f"{len(df)}", help="Number of renewable energy projects in the filtered sample")
    cols[1].metric("Installed capacity", f"{format_number(df['Capacidad_Instalada_MW'].sum())} MW", help="Total nominal power available")
    cols[2].metric("Daily generation", f"{format_number(df['Generacion_Diaria_MWh'].sum())} MWh", help="Sum of daily energy output")
    cols[3].metric("Average efficiency", f"{format_number(df['Eficiencia_Planta_Pct'].mean())} %", help="Mean plant efficiency in the current selection")
    connected_pct = df["Conectado_SIN"].mean() * 100 if not df.empty else 0
    cols[4].metric("Connected to SIN", f"{format_number(connected_pct)} %", help="Share of projects already connected to the national grid")


def render_overview(df: pd.DataFrame) -> None:
    display_section_header(
        "Overview",
        "A quick executive snapshot of the energy portfolio: size, readiness and operating profile.",
    )
    render_kpis(df)

    top_technology = df["Tecnologia"].value_counts().idxmax()
    top_technology_count = int(df["Tecnologia"].value_counts().max())
    top_state = df["Estado_Actual"].value_counts().idxmax()
    connected_ratio = df["Conectado_SIN"].mean() * 100 if not df.empty else 0
    add_insight(
        f"The most common technology is <b>{top_technology}</b> with <b>{top_technology_count}</b> projects. "
        f"The most frequent project state is <b>{top_state}</b>, and <b>{connected_ratio:.1f}%</b> of the portfolio is already connected to SIN."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Technology mix")
        fig, ax = plt.subplots(figsize=(10, 5))
        tech_order = df["Tecnologia"].value_counts().index
        sns.countplot(data=df, y="Tecnologia", order=tech_order, ax=ax, palette="Blues_r")
        ax.set_xlabel("Number of projects")
        ax.set_ylabel("Technology")
        ax.set_title("Portfolio by technology", pad=14, weight="bold")
        ax.grid(axis="x", alpha=0.2)
        for container in ax.containers:
            ax.bar_label(container, padding=4, fontsize=10)
        st.pyplot(fig, clear_figure=True)
        tech_counts = df["Tecnologia"].value_counts()
        add_insight(
            f"{tech_counts.index[0]} leads the portfolio with {tech_counts.iloc[0]} projects, which is {top_share_text(int(tech_counts.iloc[0]), len(df))} of the filtered dataset."
        )

    with right:
        st.markdown("#### Project state mix")
        state_counts = df["Estado_Actual"].value_counts().reset_index()
        state_counts.columns = ["Project state", "Count"]
        state_counts["Share"] = (state_counts["Count"] / state_counts["Count"].sum() * 100).round(1)
        fig = px.bar(
            state_counts,
            x="Count",
            y="Project state",
            orientation="h",
            title="Project states",
            text="Count",
            color="Project state",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False,
            xaxis_title="Number of projects",
            yaxis_title="",
            title_font_size=22,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        lead_state = state_counts.iloc[0]
        add_insight(
            f"<b>{lead_state['Project state']}</b> is the most common state with {int(lead_state['Count'])} projects ({lead_state['Share']:.1f}%)."
        )


def render_numeric_eda(df: pd.DataFrame) -> None:
    display_section_header(
        "Numeric EDA",
        "We describe the size, spread and balance of the measurable variables that drive the project analysis.",
    )
    numeric_cols = numeric_columns(df)
    summary = df[numeric_cols].describe().T
    summary["median_value"] = df[numeric_cols].median()
    summary["variance"] = df[numeric_cols].var()
    summary["missing"] = df[numeric_cols].isna().sum()
    summary = summary.rename(
        columns={
            "count": "count",
            "mean": "mean",
            "std": "std",
            "min": "min",
            "25%": "q1",
            "50%": "median_from_describe",
            "75%": "q3",
            "max": "max",
        }
    )
    st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)

    best_eff = df.loc[df["Eficiencia_Planta_Pct"].idxmax()]
    add_insight(
        f"The highest efficiency belongs to <b>{best_eff['ID_Proyecto']}</b> ({best_eff['Tecnologia']}) with <b>{best_eff['Eficiencia_Planta_Pct']:.1f}%</b>. "
        f"This is useful to identify which technologies are operating closer to their best performance."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Correlation heatmap")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Pearson correlation between numeric variables", pad=14, weight="bold")
        st.pyplot(fig, clear_figure=True)
        upper_triangle = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        strongest = upper_triangle.stack().sort_values(ascending=False)
        if not strongest.empty:
            pair = strongest.index[0]
            add_insight(
                f"The strongest numeric relationship is between <b>{pair[0]}</b> and <b>{pair[1]}</b> with a correlation of <b>{corr.loc[pair[0], pair[1]]:.2f}</b>."
            )

    with right:
        st.markdown("#### Distribution")
        selected_column = st.selectbox("Numeric column", numeric_cols)
        fig = px.histogram(
            df,
            x=selected_column,
            nbins=20,
            marginal="box",
            title=f"Distribution of {selected_column}",
            template="plotly_white",
            color_discrete_sequence=[PRIMARY_COLOR],
        )
        fig.update_layout(
            xaxis_title=selected_column,
            yaxis_title="Number of projects",
            title_font_size=22,
            bargap=0.1,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        median_value = df[selected_column].median()
        mean_value = df[selected_column].mean()
        add_insight(
            f"For <b>{selected_column}</b>, the mean is <b>{mean_value:.2f}</b> and the median is <b>{median_value:.2f}</b>. "
            f"If these values differ a lot, the variable is likely skewed and a few projects may be pulling the average."
        )


def render_categorical_eda(df: pd.DataFrame) -> None:
    display_section_header(
        "Categorical EDA",
        "Here we translate categories into shares so the portfolio structure is easier to read.",
    )
    categorical_cols = categorical_columns(df)
    selected_column = st.selectbox("Categorical column", categorical_cols)

    counts = df[selected_column].value_counts().reset_index()
    counts.columns = [selected_column, "Count"]
    counts["Share (%)"] = (counts["Count"] / counts["Count"].sum() * 100).round(2)

    left, right = st.columns(2)
    with left:
        st.dataframe(counts, use_container_width=True)
        add_insight(
            f"The leading category in <b>{selected_column}</b> is <b>{counts.iloc[0, 0]}</b> with <b>{counts.iloc[0, 1]}</b> projects ({counts.iloc[0, 2]:.1f}%)."
        )

    with right:
        fig = px.pie(
            counts,
            values="Count",
            names=selected_column,
            hole=0.4,
            title=f"Share of {selected_column}",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(title_font_size=22)
        st.plotly_chart(fig, use_container_width=True)


def render_data_quality(df: pd.DataFrame) -> None:
    display_section_header(
        "Data quality",
        "We surface missing values and duplicates so the chart interpretation is trustworthy.",
    )
    missing_report = missing_values_report(df)
    duplicate_rows = duplicate_count(df)

    cols = st.columns(3)
    cols[0].metric("Duplicate rows", f"{duplicate_rows}")
    cols[1].metric("Columns with missing values", f"{len(missing_report)}")
    cols[2].metric("Total missing values", f"{int(df.isna().sum().sum())}")

    if missing_report.empty:
        st.success("No missing values were found in the filtered dataset.")
    else:
        st.dataframe(missing_report, use_container_width=True)

    if duplicate_rows == 0:
        add_insight("There are no duplicate records in the current selection, so the sample looks clean for analysis.")
    else:
        add_insight(
            f"There are <b>{duplicate_rows}</b> duplicate rows. Removing them may improve the reliability of the summary statistics."
        )


def render_outliers(df: pd.DataFrame) -> None:
    display_section_header(
        "Outlier review",
        "Boxplots make extreme values visible and help explain whether a variable is unusually spread out.",
    )
    selected_column = st.selectbox(
        "Numeric variable for boxplot",
        numeric_columns(df),
        key="outlier_boxplot",
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(x=df[selected_column], ax=ax, color=SECONDARY_COLOR)
    ax.set_xlabel(selected_column)
    ax.set_title(f"Outlier detection for {selected_column}", pad=14, weight="bold")
    st.pyplot(fig, clear_figure=True)

    q1 = df[selected_column].quantile(0.25)
    q3 = df[selected_column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[selected_column] < lower) | (df[selected_column] > upper)]
    add_insight(
        f"The typical range for <b>{selected_column}</b> is between <b>{q1:.2f}</b> and <b>{q3:.2f}</b>. "
        f"Values outside <b>{lower:.2f}</b> to <b>{upper:.2f}</b> are potential outliers. We detected <b>{len(outliers)}</b> such records."
    )


def render_time_eda(df: pd.DataFrame) -> None:
    display_section_header(
        "Time-based EDA",
        "We aggregate by month to show whether capacity, generation and project count change over time.",
    )
    monthly = (
        df.assign(Month=df["Fecha_Entrada_Operacion"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month", as_index=False)
        .agg(
            Projects=("ID_Proyecto", "count"),
            Capacity_MW=("Capacidad_Instalada_MW", "sum"),
            Generation_MWh=("Generacion_Diaria_MWh", "sum"),
            Efficiency=("Eficiencia_Planta_Pct", "mean"),
        )
    )

    fig = px.line(
        monthly,
        x="Month",
        y=["Projects", "Capacity_MW", "Generation_MWh"],
        markers=True,
        title="Monthly evolution",
        template="plotly_white",
        color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR],
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Monthly total",
        title_font_size=22,
        legend_title_text="Metric",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    max_generation = monthly.loc[monthly["Generation_MWh"].idxmax()]
    add_insight(
        f"The strongest month for generation is <b>{max_generation['Month'].strftime('%Y-%m')}</b> with <b>{max_generation['Generation_MWh']:.1f} MWh</b>."
    )


def render_relationship_eda(df: pd.DataFrame) -> None:
    display_section_header(
        "Relationship analysis",
        "These charts help explain whether higher installed capacity is actually associated with higher generation.",
    )
    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            df,
            x="Capacidad_Instalada_MW",
            y="Generacion_Diaria_MWh",
            color="Tecnologia",
            size="Inversion_Inicial_MUSD",
            hover_data=["Operador", "Estado_Actual"],
            title="Capacity vs daily generation",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            xaxis_title="Installed capacity (MW)",
            yaxis_title="Daily generation (MWh)",
            title_font_size=22,
            legend_title_text="Technology",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(
            data=df,
            x="Capacidad_Instalada_MW",
            y="Generacion_Diaria_MWh",
            hue="Tecnologia",
            ax=ax,
        )
        ax.set_xlabel("Installed capacity (MW)")
        ax.set_ylabel("Daily generation (MWh)")
        ax.set_title("Capacity vs generation by technology", pad=14, weight="bold")
        st.pyplot(fig, clear_figure=True)

    corr_value = df[["Capacidad_Instalada_MW", "Generacion_Diaria_MWh"]].corr().iloc[0, 1]
    add_insight(
        f"The relationship between installed capacity and daily generation is <b>{corr_value:.2f}</b>. "
        f"Values close to 1 mean larger plants usually generate more; values near 0 mean the link is weak."
    )


def render_portfolio_story(df: pd.DataFrame) -> None:
    display_section_header(
        "Portfolio story",
        "A short interpretation that helps non-technical readers understand the sample at a glance.",
    )
    total_capacity = df["Capacidad_Instalada_MW"].sum()
    total_generation = df["Generacion_Diaria_MWh"].sum()
    mean_efficiency = df["Eficiencia_Planta_Pct"].mean()
    most_common_tech = df["Tecnologia"].value_counts().idxmax()
    most_common_state = df["Estado_Actual"].value_counts().idxmax()

    add_insight(
        f"This portfolio contains <b>{len(df)}</b> projects, <b>{format_number(total_capacity)}</b> MW of installed capacity and <b>{format_number(total_generation)}</b> MWh of daily generation. "
        f"The average efficiency is <b>{mean_efficiency:.1f}%</b>. The dominant technology is <b>{most_common_tech}</b> and the most frequent stage is <b>{most_common_state}</b>."
    )


def render_raw_data(df: pd.DataFrame) -> None:
    st.subheader("Filtered data")
    st.dataframe(df, use_container_width=True)
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv_data,
        file_name="filtered_renewable_energy_eda.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Renewable Energy Exploratory Data Analysis")
    st.caption(
        "Clean-code Streamlit dashboard built with Plotly, Seaborn and Matplotlib. "
        "The goal is to make the data easier to read, compare and explain."
    )

    base_df = clean_data(load_data())
    filters = build_sidebar(base_df)

    dataset = clean_data(load_data(filters["uploaded_file"]))
    filtered_df = filter_data(dataset, filters)

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        return

    st.write(f"Filtered records: **{len(filtered_df)}**")
    render_portfolio_story(filtered_df)

    tab_overview, tab_numeric, tab_categorical, tab_quality, tab_outliers, tab_time, tab_relationships, tab_raw = st.tabs(
        [
            "Overview",
            "Numeric",
            "Categorical",
            "Quality",
            "Outliers",
            "Time",
            "Relationships",
            "Raw data",
        ]
    )

    with tab_overview:
        render_overview(filtered_df)

    with tab_numeric:
        render_numeric_eda(filtered_df)

    with tab_categorical:
        render_categorical_eda(filtered_df)

    with tab_quality:
        render_data_quality(filtered_df)

    with tab_outliers:
        render_outliers(filtered_df)

    with tab_time:
        render_time_eda(filtered_df)

    with tab_relationships:
        render_relationship_eda(filtered_df)

    with tab_raw:
        render_raw_data(filtered_df)


if __name__ == "__main__":
    main()
