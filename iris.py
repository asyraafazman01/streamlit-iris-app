import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Iris Data Explorer",
    page_icon="🌸",
    layout="wide"
)

# ---------------- Data loading ----------------
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        },
        inplace=True,
    )
    df["species"] = df["target"].map(dict(zip(range(3), iris.target_names)))
    return df

df = load_data()

# ---------------- Sidebar filters ----------------
st.sidebar.title("Filters")

species_options = sorted(df["species"].unique())
selected_species = st.sidebar.multiselect(
    "Select species",
    species_options,
    default=species_options,
)

sepal_min, sepal_max = float(df["sepal_length"].min()), float(df["sepal_length"].max())
sepal_range = st.sidebar.slider(
    "Filter by sepal length (cm)",
    min_value=round(sepal_min, 1),
    max_value=round(sepal_max, 1),
    value=(round(sepal_min, 1), round(sepal_max, 1)),
    step=0.1,
)

show_raw = st.sidebar.checkbox("Show raw filtered data")

# Apply filters
filtered_df = df[
    (df["species"].isin(selected_species)) &
    (df["sepal_length"] >= sepal_range[0]) &
    (df["sepal_length"] <= sepal_range[1])
]

# ---------------- Title & description ----------------
st.title("🌸 Iris Data Explorer")


st.caption(f"Number of rows after filtering: **{len(filtered_df)}**")

# ---------------- Data summary (Task requirement) ----------------
st.subheader("Data Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Number of samples",
    len(filtered_df),
)

col2.metric(
    "Average sepal length (cm)",
    f"{filtered_df['sepal_length'].mean():.2f}" if not filtered_df.empty else "N/A",
)

col3.metric(
    "Average petal length (cm)",
    f"{filtered_df['petal_length'].mean():.2f}" if not filtered_df.empty else "N/A",
)

if show_raw:
    st.write("### Filtered Data")
    st.dataframe(filtered_df, use_container_width=True)

# ---------------- Visualizations (at least 2 types) ----------------
st.subheader("Visualizations")

tab1, tab2 = st.tabs(["Scatter Plot", "Histogram"])

# Scatter plot: petal vs sepal (Plotly)
with tab1:
    st.markdown("**Scatter Plot – Sepal Length vs Petal Length**")

    if filtered_df.empty:
        st.warning("No data to display. Try relaxing your filters in the sidebar.")
    else:
        fig_scatter = px.scatter(
            filtered_df,
            x="sepal_length",
            y="petal_length",
            color="species",
            size="sepal_width",
            hover_data=["sepal_width", "petal_width"],
            labels={
                "sepal_length": "Sepal Length (cm)",
                "petal_length": "Petal Length (cm)",
            },
            title="Sepal Length vs Petal Length by Species",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# Histogram: distribution of sepal length
with tab2:
    st.markdown("**Histogram – Distribution of Sepal Length**")

    if filtered_df.empty:
        st.warning("No data to display. Try relaxing your filters in the sidebar.")
    else:
        fig_hist = px.histogram(
            filtered_df,
            x="sepal_length",
            color="species",
            nbins=20,
            labels={"sepal_length": "Sepal Length (cm)"},
            title="Distribution of Sepal Length by Species",
            barmode="overlay",
            opacity=0.7,
        )
        st.plotly_chart(fig_hist, use_container_width=True)


