"""Streamlit EDA application for renewable energy projects."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st


APP_TITLE = "Renewable Energy EDA"
DATA_FILE = Path(__file__).with_name("energia_renovable.csv")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid", context="talk")


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
    cols[0].metric("Projects", f"{len(df)}")
    cols[1].metric("Installed capacity", f"{df['Capacidad_Instalada_MW'].sum():,.1f} MW")
    cols[2].metric("Daily generation", f"{df['Generacion_Diaria_MWh'].sum():,.1f} MWh")
    cols[3].metric("Average efficiency", f"{df['Eficiencia_Planta_Pct'].mean():,.1f} %")
    connected_pct = df["Conectado_SIN"].mean() * 100 if not df.empty else 0
    cols[4].metric("Connected to SIN", f"{connected_pct:,.1f} %")


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Overview")
    render_kpis(df)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Technology mix")
        fig, ax = plt.subplots(figsize=(10, 5))
        tech_order = df["Tecnologia"].value_counts().index
        sns.countplot(data=df, y="Tecnologia", order=tech_order, ax=ax, palette="Blues_r")
        ax.set_xlabel("Count")
        ax.set_ylabel("Technology")
        st.pyplot(fig, clear_figure=True)

    with right:
        st.markdown("#### Project state mix")
        state_counts = df["Estado_Actual"].value_counts().reset_index()
        state_counts.columns = ["Project state", "Count"]
        fig = px.bar(
            state_counts,
            x="Count",
            y="Project state",
            orientation="h",
            title="Project states",
            text="Count",
            color="Project state",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_numeric_eda(df: pd.DataFrame) -> None:
    st.subheader("Numeric EDA")
    numeric_cols = numeric_columns(df)
    summary = df[numeric_cols].describe().T
    summary["median"] = df[numeric_cols].median()
    summary["variance"] = df[numeric_cols].var()
    summary["missing"] = df[numeric_cols].isna().sum()
    st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Correlation heatmap")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
        st.pyplot(fig, clear_figure=True)

    with right:
        st.markdown("#### Distribution")
        selected_column = st.selectbox("Numeric column", numeric_cols)
        fig = px.histogram(
            df,
            x=selected_column,
            nbins=20,
            marginal="box",
            title=f"Distribution of {selected_column}",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_categorical_eda(df: pd.DataFrame) -> None:
    st.subheader("Categorical EDA")
    categorical_cols = categorical_columns(df)
    selected_column = st.selectbox("Categorical column", categorical_cols)

    counts = df[selected_column].value_counts().reset_index()
    counts.columns = [selected_column, "Count"]
    counts["Share (%)"] = (counts["Count"] / counts["Count"].sum() * 100).round(2)

    left, right = st.columns(2)
    with left:
        st.dataframe(counts, use_container_width=True)

    with right:
        fig = px.pie(
            counts,
            values="Count",
            names=selected_column,
            hole=0.4,
            title=f"Share of {selected_column}",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_data_quality(df: pd.DataFrame) -> None:
    st.subheader("Data quality")
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


def render_outliers(df: pd.DataFrame) -> None:
    st.subheader("Outlier review")
    selected_column = st.selectbox(
        "Numeric variable for boxplot",
        numeric_columns(df),
        key="outlier_boxplot",
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(x=df[selected_column], ax=ax, color="#2a9d8f")
    ax.set_xlabel(selected_column)
    st.pyplot(fig, clear_figure=True)


def render_time_eda(df: pd.DataFrame) -> None:
    st.subheader("Time-based EDA")
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
    )
    st.plotly_chart(fig, use_container_width=True)


def render_relationship_eda(df: pd.DataFrame) -> None:
    st.subheader("Relationship analysis")
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
        st.pyplot(fig, clear_figure=True)


def main() -> None:
    st.title("Renewable Energy Exploratory Data Analysis")
    st.caption("Clean-code Streamlit dashboard built with Plotly, Seaborn, Matplotlib and a local cyborn helper module.")

    base_df = clean_data(load_data())
    filters = build_sidebar(base_df)

    dataset = clean_data(load_data(filters["uploaded_file"]))
    filtered_df = filter_data(dataset, filters)

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        return

    st.write(f"Filtered records: **{len(filtered_df)}**")

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
        st.dataframe(filtered_df, use_container_width=True)


if __name__ == "__main__":
    main()
