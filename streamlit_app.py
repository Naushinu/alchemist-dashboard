import streamlit as st
import pandas as pd
import plotly.express as px
import os
from alchemist_analysis import (
    df,
    plot_histograms, plot_sentence_category_bar,
    plot_rolling_compound, plot_segment_chart,
    add_text2emotion,
    add_nrc_emotions, plot_nrc_distribution,
    print_translation_note
)

import plotly.graph_objects as go

# Load location data
loc_df = pd.read_csv("Alchemist_location.csv")
loc_df_clean = loc_df[loc_df["Order"] != 4].sort_values("Order").reset_index(drop=True)

# Build journey map function
def plot_journey_map():
    fig = go.Figure()

    # Unique colors for each stop
    color_palette = px.colors.qualitative.Plotly  # or pick your own hex codes
    num_locations = len(loc_df_clean)

    # Add location markers with individual colors
    for i, row in loc_df_clean.iterrows():
        fig.add_trace(go.Scattergeo(
            lon=[row["Longitude"]],
            lat=[row["Latitude"]],
            mode="markers",
            marker=dict(
                size=10,
                color=color_palette[i % len(color_palette)],
                line=dict(width=1, color="black")
            ),
            name=row["Location"],  # Legend label
            text=row["Description"],
            hoverinfo="text"
        ))

    # Add travel lines
    for i in range(len(loc_df_clean) - 1):
        a = loc_df_clean.iloc[i]
        b = loc_df_clean.iloc[i + 1]

        desert_hop = a["Order"] == 3 and b["Order"] == 5

        if desert_hop:
            fig.add_trace(go.Scattergeo(
                lon=[a["Longitude"], b["Longitude"]],
                lat=[a["Latitude"], b["Latitude"]],
                mode="lines",
                line=dict(width=2, color="sandybrown", dash="dot"),
                name="Journey across Sahara Desert"
            ))
        else:
            fig.add_trace(go.Scattergeo(
                lon=[a["Longitude"], b["Longitude"]],
                lat=[a["Latitude"], b["Latitude"]],
                mode="lines",
                line=dict(width=2, color="gray", dash="solid"),
                showlegend=False
            ))

    # Map layout styling
    fig.update_geos(
        showcountries=True, countrycolor="black",
        landcolor="rgb(230,220,200)",
        oceancolor="lightblue", showocean=True,
        showland=True, showlakes=True, lakecolor="lightblue",
        resolution=50,
        lataxis_range=[15, 50],  # expanded vertically
        lonaxis_range=[-15, 40]  # expanded horizontally
    )

    fig.update_layout(
        title="Santiago’s Journey in *The Alchemist*",
        margin=dict(r=0, t=50, l=0, b=0),
        legend=dict(
            title="Stops",
            bgcolor="white",
            bordercolor="black",
            borderwidth=0.5
        )
    )

    return fig

# ---- Streamlit Setup ----
st.set_page_config(page_title="Textual Sentiment in The Alchemist", layout="wide")
st.title("Textual Sentiment in The Alchemist")

with st.expander("📌 Project Overview"):
    print_translation_note()
    st.markdown("""
    **Project Summary:**  
    This dashboard offers an exploratory look into *The Alchemist* by Paulo Coelho using natural language processing tools. My goal is to visualize sentiment patterns, emotional themes, and the physical journey of the protagonist (Santiago) throughout the novel.
    
    **Included Analyses:**
    1. **Sentiment Analysis** using VADER: sentence-level polarity (positive/negative/neutral) with rolling trends and segment summaries.
    2. **Emotion Analysis** using two tools:
        - `text2emotion`: coarse-grained emotions like Happy, Sad, Fear, etc.
        - `NRC EmoLex`: fine-grained categories like Trust, Anticipation, Disgust.
    3. **Journey Mapping**: a geographic map tracing Santiago’s physical journey from Spain to Egypt.

    **Limitations:**
    - An **English translation** of a Portuguese book is used, therefore linguistic nuances may be lost.
    - These interpretations are approximate and serve as **exploratory tools**, not definitive literary conclusions.    
    """)

# ---- Sidebar navigation ----
analysis_type = st.sidebar.radio("Select View", [
    "📈 Sentiment Trends", "🎭 Emotion Analysis", "🗺️ Santiago's Journey Map"
])

if analysis_type == "📈 Sentiment Trends":
    st.subheader("VADER Polarity Distributions")
    plot_histograms()

    st.subheader("Sentence Category Count")
    plot_sentence_category_bar()

    st.subheader("Rolling Sentiment Trend")
    plot_rolling_compound()

    st.subheader("Sentiment by Book Segment")
    plot_segment_chart()

elif analysis_type == "🎭 Emotion Analysis":
    st.subheader("Using Text2Emotion:")

    with st.spinner("We're processing the text2emotion analysis...this could take us a moment."):
        try:
            df_t2e = add_text2emotion()

            t2e_totals = df_t2e[["Happy", "Angry", "Surprise", "Sad", "Fear"]].sum().reset_index()
            t2e_totals.columns = ["Emotion", "Score"]

            fig = px.bar(
                t2e_totals,
                x="Emotion",
                y="Score",
                title="Text2Emotion – Emotion Distribution",
                color_discrete_sequence=["#719ba6"]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.success("Text2Emotion analysis completed!")

        except Exception as e:
            st.error(f"Error in Text2Emotion analysis: {str(e)}")

    st.subheader("Using NRC Emotion Lexicon:")

    with st.spinner("We're processing the NRC emotion analysis...this could take us a moment."):
        try:
            df_nrc = add_nrc_emotions()
            plot_nrc_distribution(df_nrc)
            st.success("NRC Emotion analysis completed!")
        except Exception as e:
            st.error(f"Error in NRC Emotion analysis: {str(e)}")

elif analysis_type == "🗺️ Santiago's Journey Map":
    st.subheader("Map of Santiago’s Journey")
    st.plotly_chart(plot_journey_map(), use_container_width=True)