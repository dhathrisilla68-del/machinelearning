
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RideRush | Smart Mobility",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "final_features.parquet"
MODEL_FILE = BASE_DIR / "models" / "tuned_random_forest_model.pkl"
MODEL_RESULTS_FILE = BASE_DIR / "results" / "model_comparison.csv"
TUNED_RESULTS_FILE = BASE_DIR / "results" / "tuned_model_results.csv"
FEATURE_FILE = BASE_DIR / "results" / "feature_importance.csv"
CLUSTER_FILE = BASE_DIR / "data" / "clustering_results.parquet"


# =========================================================
# 2. PREMIUM WEBSITE DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f4f7fd;
    color: #172747;
}

[data-testid="stHeader"] {
    background: rgba(244,247,253,0.95);
}

.block-container {
    max-width: 1550px;
    padding-top: 1.8rem;
    padding-bottom: 2.8rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101c38 0%, #172c53 100%);
    border-right: 1px solid #263d64;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] * {
    color: #eaf1ff;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #aabbd9;
}

[data-testid="stSidebar"] hr {
    border-color: #304469;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 7px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 9px 10px;
    border-radius: 10px;
}

h1, h2, h3 {
    color: #152849;
    font-weight: 800;
    letter-spacing: -0.6px;
}

.rr-logo {
    font-size: 29px;
    font-weight: 800;
    color: #65a2ff;
    letter-spacing: -1.5px;
}

.rr-logo span {
    color: #ffffff;
}

.rr-tagline {
    font-size: 10px;
    color: #aabbd9;
    letter-spacing: 1.7px;
    margin-top: 3px;
    margin-bottom: 25px;
}

.rr-hero {
    background: linear-gradient(115deg, #101e3d 0%, #174fbd 58%, #438cff 100%);
    border-radius: 23px;
    padding: 33px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 15px 38px rgba(23, 80, 180, 0.19);
}

.rr-hero h1 {
    color: #ffffff !important;
    font-size: 32px;
    margin: 13px 0 8px 0;
    font-weight: 800;
}

.rr-hero p {
    color: #e1ebff !important;
    font-size: 13px;
    margin-bottom: 0;
    line-height: 1.7;
}

.rr-pill {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.24);
    border-radius: 50px;
    padding: 7px 12px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}

.rr-card {
    background: #ffffff;
    border: 1px solid #e5ebf6;
    border-radius: 18px;
    padding: 21px;
    box-shadow: 0 7px 25px rgba(22, 48, 91, 0.045);
    margin-bottom: 14px;
}

.rr-card-label {
    color: #7d8ba5;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    margin-bottom: 12px;
}

.rr-card-value {
    color: #152849;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
}

.rr-card-note {
    color: #138667;
    font-size: 11px;
    margin-top: 8px;
    font-weight: 600;
}

.rr-section {
    color: #152849;
    font-size: 19px;
    font-weight: 800;
    margin: 19px 0 5px 0;
}

.rr-muted {
    color: #8390a7;
    font-size: 12px;
    margin-bottom: 14px;
}

.rr-status {
    display: inline-block;
    background: #d8f8e9;
    color: #087b57 !important;
    border-radius: 30px;
    padding: 7px 11px;
    font-size: 10px;
    font-weight: 800;
}

.rr-result {
    background: linear-gradient(145deg, #102043, #1a56bf 72%, #3d87ff);
    border-radius: 22px;
    padding: 27px;
    color: white;
    box-shadow: 0 12px 28px rgba(23, 80, 180, 0.18);
}

.rr-result-label {
    color: #dce9ff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

.rr-result-number {
    font-size: 43px;
    color: #ffffff;
    font-weight: 800;
    margin: 12px 0;
}

.rr-result-note {
    color: #dce9ff;
    font-size: 11px;
    line-height: 1.7;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(105deg, #1769ff, #438cff);
    color: white;
    border: none;
    border-radius: 11px;
    min-height: 44px;
    font-weight: 700;
    box-shadow: 0 5px 15px rgba(23,105,255,0.16);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #1259df;
    color: white;
    border: none;
}

div[data-baseweb="select"] > div,
.stNumberInput input {
    border-radius: 10px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5ebf6;
    border-radius: 16px;
    padding: 17px;
    box-shadow: 0 5px 20px rgba(22,48,91,0.035);
}

div[data-testid="stMetricLabel"] {
    color: #7d8ba5;
}

div[data-testid="stMetricValue"] {
    color: #152849;
    font-weight: 800;
}

[data-testid="stDataFrame"] {
    border: 1px solid #e5ebf6;
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: #e5ebf6;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD DATA AND MODEL
# =========================================================

@st.cache_data(show_spinner="Loading RideRush data...")
def load_table(path_string):
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)

        return pd.read_csv(path)

    except Exception:
        return pd.DataFrame()


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(path_string):
    path = Path(path_string)

    if not path.exists():
        return None

    try:
        return joblib.load(path)

    except Exception:
        return None


df = load_table(str(DATA_FILE))
model = load_model(str(MODEL_FILE))
model_results = load_table(str(MODEL_RESULTS_FILE))
tuned_results = load_table(str(TUNED_RESULTS_FILE))
feature_df = load_table(str(FEATURE_FILE))
cluster_df = load_table(str(CLUSTER_FILE))


# =========================================================
# 4. REUSABLE FUNCTIONS
# =========================================================

def section(title, subtitle=""):
    st.markdown(
        f'<div class="rr-section">{title}</div>'
        f'<div class="rr-muted">{subtitle}</div>',
        unsafe_allow_html=True
    )


def hero(title, subtitle, label):
    st.markdown(f"""
    <div class="rr-hero">
        <span class="rr-pill">{label}</span>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, note, emoji):
    st.markdown(f"""
    <div class="rr-card">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div class="rr-card-label">{label}</div>
            <div style="font-size:22px">{emoji}</div>
        </div>
        <div class="rr-card-value">{value}</div>
        <div class="rr-card-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def show_missing(path):
    st.warning(
        f"File not found: `{path}`. "
        "Please check the RideRush project folders."
    )


def find_column(data, candidates):
    for col in candidates:
        if col in data.columns:
            return col
    return None


def styled_chart(fig, height=340):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=50, b=15),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font=dict(
            family="Inter",
            size=11,
            color="#6d7b93"
        ),
        title_font=dict(
            family="Inter",
            size=16,
            color="#152849"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#e8edf5",
        zeroline=False
    )

    fig.update_yaxes(
        gridcolor="#edf1f8",
        zeroline=False
    )

    return fig


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="rr-logo">Ride<span>Rush</span> 🚕</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="rr-tagline">SMART MOBILITY ANALYTICS</div>',
        unsafe_allow_html=True
    )

    st.markdown("##### WORKSPACE")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Demand Analysis",
            "ETA Prediction",
            "Model Evaluation",
            "Feature Importance",
            "Clustering"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("##### SYSTEM STATUS")

    if model is not None:
        st.markdown(
            '<span class="rr-status">● MODEL ONLINE</span>',
            unsafe_allow_html=True
        )
    else:
        st.warning("Model file not loaded")

    st.markdown("")
    st.caption("PROJECT  /  RideRush")
    st.caption("DATA  /  NYC Yellow Taxi")
    st.caption("PERIOD  /  May 2026")
    st.caption("ML LIFECYCLE  /  M1–M6")


# =========================================================
# 6. OVERVIEW PAGE
# =========================================================

if page == "Overview":

    hero(
        "Welcome to RideRush",
        "Your intelligent mobility analytics platform. "
        "Explore trip demand, discover travel patterns, and predict trip duration.",
        "✦ AI-POWERED MOBILITY PLATFORM"
    )

    if df.empty:
        show_missing(DATA_FILE)

    else:
        total_trips = len(df)

        avg_duration = (
            df["trip_duration"].mean()
            if "trip_duration" in df.columns else None
        )

        avg_distance = (
            df["trip_distance"].mean()
            if "trip_distance" in df.columns else None
        )

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            metric_card(
                "TOTAL TRIPS",
                f"{total_trips:,}",
                "Processed trip records",
                "🚕"
            )

        with k2:
            metric_card(
                "AVG. DURATION",
                f"{avg_duration:.1f} min" if avg_duration is not None else "N/A",
                "Recorded trip duration",
                "⏱️"
            )

        with k3:
            metric_card(
                "AVG. DISTANCE",
                f"{avg_distance:.2f}" if avg_distance is not None else "N/A",
                "Dataset distance units",
                "📍"
            )

        with k4:
            metric_card(
                "PREDICTION MODEL",
                "Ready" if model is not None else "Unavailable",
                "Saved ML model status",
                "🤖"
            )

        left, right = st.columns([1.65, 1])

        with left:

            section(
                "Hourly Demand Trends",
                "Discover trip activity throughout the day"
            )

            if "hour" in df.columns:

                hourly = (
                    df.groupby("hour")
                    .size()
                    .reindex(range(24), fill_value=0)
                    .rename("Trips")
                    .reset_index()
                )

                hourly.columns = ["Hour", "Trips"]

                fig = px.area(
                    hourly,
                    x="Hour",
                    y="Trips",
                    markers=True,
                    color_discrete_sequence=["#2674ff"]
                )

                fig.update_traces(
                    line=dict(width=3),
                    fillcolor="rgba(38,116,255,0.13)"
                )

                st.plotly_chart(
                    styled_chart(fig),
                    width="stretch"
                )

            else:
                st.info("The dataset does not contain an hour column.")

        with right:

            section(
                "Rush Hour Overview",
                "Compare rush-hour and other trips"
            )

            if "is_rush_hour" in df.columns:

                rush_counts = (
                    df["is_rush_hour"]
                    .map({1: "Rush Hour", 0: "Other Hours"})
                    .fillna("Other Hours")
                    .value_counts()
                )

                fig = go.Figure(
                    data=[go.Pie(
                        labels=rush_counts.index,
                        values=rush_counts.values,
                        hole=0.72,
                        marker=dict(
                            colors=["#2674ff", "#dce7fb"],
                            line=dict(color="white", width=4)
                        ),
                        textinfo="percent",
                        textfont=dict(size=12)
                    )]
                )

                fig.update_layout(
                    annotations=[dict(
                        text="TRIPS",
                        x=0.5,
                        y=0.5,
                        font_size=13,
                        font_color="#152849",
                        showarrow=False
                    )]
                )

                st.plotly_chart(
                    styled_chart(fig),
                    width="stretch"
                )

            else:
                st.info("Rush-hour information is unavailable.")

        section(
            "Trip Dataset Preview",
            "A quick look at the processed dataset"
        )

        st.dataframe(df.head(10), width="stretch")

        st.markdown("""
        <div class="rr-card">
            <div class="rr-card-label">MACHINE LEARNING WORKFLOW</div>
            <p style="color:#152849;font-weight:600">
                01 &nbsp; Data Cleaning
                &nbsp; → &nbsp; 02 &nbsp; Feature Engineering
                &nbsp; → &nbsp; 03 &nbsp; Model Training
                &nbsp; → &nbsp; 04 &nbsp; Evaluation
                &nbsp; → &nbsp; 05 &nbsp; Prediction
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# 7. DEMAND ANALYSIS PAGE
# =========================================================

elif page == "Demand Analysis":

    hero(
        "Demand Intelligence",
        "Understand when taxi trips are most frequent using historical data.",
        "📊 HISTORICAL TRIP ANALYTICS"
    )

    if df.empty:
        show_missing(DATA_FILE)

    else:

        c1, c2 = st.columns(2)

        with c1:

            section("Hourly Demand", "Trips grouped by pickup hour")

            if "hour" in df.columns:

                hourly = (
                    df.groupby("hour")
                    .size()
                    .reindex(range(24), fill_value=0)
                    .rename("Trips")
                    .reset_index()
                )

                hourly.columns = ["Hour", "Trips"]

                fig = px.bar(
                    hourly,
                    x="Hour",
                    y="Trips",
                    color="Trips",
                    color_continuous_scale=["#d9e7ff", "#1769ff"]
                )

                fig.update_layout(coloraxis_showscale=False)

                st.plotly_chart(
                    styled_chart(fig),
                    width="stretch"
                )

            else:
                st.info("Hour feature unavailable.")

        with c2:

            section("Weekly Activity", "Trips grouped by weekday")

            if "day_of_week" in df.columns:

                daily = (
                    df.groupby("day_of_week")
                    .size()
                    .reindex(range(7), fill_value=0)
                    .rename("Trips")
                    .reset_index()
                )

                daily["Day"] = daily["day_of_week"].map({
                    0: "Monday",
                    1: "Tuesday",
                    2: "Wednesday",
                    3: "Thursday",
                    4: "Friday",
                    5: "Saturday",
                    6: "Sunday"
                })

                fig = px.bar(
                    daily,
                    x="Day",
                    y="Trips",
                    color_discrete_sequence=["#2674ff"]
                )

                st.plotly_chart(
                    styled_chart(fig),
                    width="stretch"
                )

            else:
                st.info("Day-of-week feature unavailable.")

        section(
            "Rush Hour Comparison",
            "Compare recorded trips during rush and non-rush hours"
        )

        if "is_rush_hour" in df.columns:

            rush = (
                df["is_rush_hour"]
                .map({1: "Rush Hour", 0: "Other Hours"})
                .fillna("Other Hours")
                .value_counts()
                .rename_axis("Period")
                .reset_index(name="Trips")
            )

            fig = px.pie(
                rush,
                names="Period",
                values="Trips",
                hole=0.62,
                color="Period",
                color_discrete_map={
                    "Rush Hour": "#1769ff",
                    "Other Hours": "#dce7fb"
                }
            )

            st.plotly_chart(
                styled_chart(fig, 350),
                width="stretch"
            )

        st.info(
            "These charts describe historical trip patterns. "
            "They do not represent live taxi demand."
        )


# =========================================================
# 8. ETA PREDICTION PAGE
# =========================================================

elif page == "ETA Prediction":

    hero(
        "Trip Duration Prediction",
        "Enter trip details and use your trained Random Forest model "
        "to estimate trip duration.",
        "🤖 MACHINE LEARNING PREDICTION"
    )

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:

        section(
            "Your Trip Details",
            "Complete the fields below"
        )

        if model is None:

            show_missing(MODEL_FILE)

        else:

            with st.form("eta_prediction_form"):

                distance = st.number_input(
                    "Trip Distance",
                    min_value=0.1,
                    max_value=100.0,
                    value=5.0,
                    step=0.5
                )

                passengers = st.number_input(
                    "Passenger Count",
                    min_value=1,
                    max_value=8,
                    value=1,
                    step=1
                )

                hour = st.slider(
                    "Pickup Hour",
                    min_value=0,
                    max_value=23,
                    value=17
                )

                day = st.slider(
                    "Day of Month",
                    min_value=1,
                    max_value=31,
                    value=15
                )

                month = st.selectbox(
                    "Month",
                    list(range(1, 13)),
                    index=4
                )

                day_of_week = st.selectbox(
                    "Day of Week",
                    list(range(7)),
                    index=3,
                    format_func=lambda x: [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday"
                    ][x]
                )

                submitted = st.form_submit_button(
                    "✨ Predict Trip Duration",
                    width="stretch"
                )

            if submitted:

                is_weekend = int(day_of_week >= 5)

                is_rush_hour = int(
                    hour in [7, 8, 9, 16, 17, 18, 19]
                )

                input_df = pd.DataFrame([{
                    "trip_distance": distance,
                    "passenger_count": passengers,
                    "hour": hour,
                    "day": day,
                    "month": month,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "is_rush_hour": is_rush_hour
                }])

                try:

                    expected_features = getattr(
                        model,
                        "feature_names_in_",
                        None
                    )

                    if expected_features is not None:

                        input_df = input_df.reindex(
                            columns=list(expected_features),
                            fill_value=0
                        )

                    prediction = float(model.predict(input_df)[0])

                    st.session_state["rr_prediction"] = max(
                        0.0, prediction
                    )

                    st.session_state["rr_prediction_distance"] = distance

                except Exception as exc:

                    st.error(
                        "Prediction failed. Check the model's training "
                        "feature names and order."
                    )

                    st.code(str(exc))

    with right:

        section(
            "Your Prediction",
            "Estimated trip duration from the saved model"
        )

        if "rr_prediction" in st.session_state:

            pred = st.session_state["rr_prediction"]

            st.markdown(f"""
            <div class="rr-result">
                <div class="rr-result-label">ESTIMATED DURATION</div>
                <div class="rr-result-number">
                    {pred:.1f}
                    <span style="font-size:17px;font-weight:500">min</span>
                </div>
                <div class="rr-result-note">
                    Prediction generated by your trained ML model.
                    Actual duration can differ due to traffic and road conditions.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            st.metric(
                "Input Trip Distance",
                f"{st.session_state['rr_prediction_distance']:.1f}"
            )

        else:

            st.markdown("""
            <div class="rr-card" style="text-align:center;padding:34px 20px">
                <div style="font-size:45px">⏱️</div>
                <h3>Ready to Predict?</h3>
                <p>Enter trip details and click the prediction button.</p>
            </div>
            """, unsafe_allow_html=True)

        section("How It Works", "Three simple steps")

        st.markdown("""
        <div class="rr-card">
            <p style="color:#1769ff;font-weight:700">01 — Enter trip details</p>
            <p style="color:#8290a5;font-size:12px">
                Provide distance, passengers, and time information.
            </p>
            <hr>
            <p style="color:#1769ff;font-weight:700">02 — Run the ML model</p>
            <p style="color:#8290a5;font-size:12px">
                Send the features to your saved prediction model.
            </p>
            <hr>
            <p style="color:#1769ff;font-weight:700">03 — View the estimate</p>
            <p style="color:#8290a5;font-size:12px">
                Display the predicted trip duration in minutes.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "This is an estimated duration, not a live GPS or traffic ETA."
        )


# =========================================================
# 9. MODEL EVALUATION PAGE
# =========================================================

elif page == "Model Evaluation":

    hero(
        "Model Performance",
        "Compare trained models and review your saved evaluation results.",
        "📈 MODEL EVALUATION"
    )

    if not model_results.empty:

        section(
            "Model Comparison",
            "Results from your model evaluation script"
        )

        st.dataframe(model_results, width="stretch")

        name_col = find_column(
            model_results,
            ["model", "Model", "model_name", "Model Name"]
        )

        metric_col = find_column(
            model_results,
            ["MAE", "mae", "RMSE", "rmse", "R2", "r2", "R²"]
        )

        if name_col and metric_col:

            chart_df = model_results.copy()

            chart_df[metric_col] = pd.to_numeric(
                chart_df[metric_col],
                errors="coerce"
            )

            chart_df = chart_df.dropna(subset=[metric_col])

            if not chart_df.empty:

                fig = px.bar(
                    chart_df,
                    x=name_col,
                    y=metric_col,
                    color_discrete_sequence=["#2674ff"],
                    title=f"Model Comparison — {metric_col}"
                )

                st.plotly_chart(
                    styled_chart(fig),
                    width="stretch"
                )

    else:

        show_missing(MODEL_RESULTS_FILE)

    section(
        "Tuned Random Forest Results",
        "Results saved after hyperparameter tuning"
    )

    if not tuned_results.empty:

        st.dataframe(tuned_results, width="stretch")

    else:

        show_missing(TUNED_RESULTS_FILE)

    st.info(
        "MAE and RMSE measure prediction error; lower is generally better. "
        "R² measures how much variation is explained by the model; "
        "higher is generally better."
    )


# =========================================================
# 10. FEATURE IMPORTANCE PAGE
# =========================================================

elif page == "Feature Importance":

    hero(
        "Feature Importance",
        "Explore which input features contribute most to the trained model.",
        "🔍 MODEL INTERPRETABILITY"
    )

    if feature_df.empty:

        show_missing(FEATURE_FILE)

    else:

        feature_col = find_column(
            feature_df,
            ["feature", "Feature", "feature_name"]
        )

        importance_col = find_column(
            feature_df,
            ["importance", "Importance", "feature_importance"]
        )

        if feature_col and importance_col:

            plot_df = feature_df.copy()

            plot_df[importance_col] = pd.to_numeric(
                plot_df[importance_col],
                errors="coerce"
            )

            plot_df = plot_df.dropna(subset=[importance_col])

            plot_df = plot_df.sort_values(
                importance_col,
                ascending=True
            )

            section(
                "Feature Ranking",
                "Features with larger importance values have greater influence "
                "on the fitted model's predictions."
            )

            fig = px.bar(
                plot_df,
                x=importance_col,
                y=feature_col,
                orientation="h",
                color=importance_col,
                color_continuous_scale=["#d8e6ff", "#1769ff"]
            )

            fig.update_layout(coloraxis_showscale=False)

            st.plotly_chart(
                styled_chart(fig, 430),
                width="stretch"
            )

            section("Detailed Importance Values")

            st.dataframe(
                plot_df.sort_values(
                    importance_col,
                    ascending=False
                ),
                width="stretch"
            )

        else:

            st.warning(
                "Could not identify the feature and importance columns. "
                "Check the column headers in feature_importance.csv."
            )


# =========================================================
# 11. CLUSTERING PAGE
# =========================================================

elif page == "Clustering":

    hero(
        "Trip Clustering",
        "Explore the groups generated by your unsupervised learning pipeline.",
        "🧩 UNSUPERVISED LEARNING"
    )

    if cluster_df.empty:

        show_missing(CLUSTER_FILE)

    else:

        section(
            "Clustering Overview",
            "Review the saved clustering output"
        )

        st.metric(
            "Records Available",
            f"{len(cluster_df):,}"
        )

        cluster_col = find_column(
            cluster_df,
            [
                "cluster",
                "Cluster",
                "kmeans_cluster",
                "cluster_label",
                "KMeans_Cluster"
            ]
        )

        if cluster_col:

            counts = (
                cluster_df[cluster_col]
                .astype(str)
                .value_counts()
                .rename_axis("Cluster")
                .reset_index(name="Trips")
            )

            left, right = st.columns(2)

            with left:

                section(
                    "Cluster Sizes",
                    "Number of records assigned to each cluster"
                )

                fig = px.bar(
                    counts,
                    x="Cluster",
                    y="Trips",
                    color_discrete_sequence=["#2674ff"]
                )

                st.plotly_chart(
                    styled_chart(fig),
                    width="stretch"
                )

            with right:

                section(
                    "Cluster Distribution",
                    "Share of records in each cluster"
                )

                fig = px.pie(
                    counts,
                    names="Cluster",
                    values="Trips",
                    hole=0.62,
                    color_discrete_sequence=px.colors.sequential.Blues
                )

                st.plotly_chart(
                    styled_chart(fig),
                    width="stretch"
                )

        else:

            st.info(
                "No recognized cluster-label column was found. "
                "Check the column names in clustering_results.parquet."
            )

        section(
            "Clustering Records",
            "Preview of the saved clustering results"
        )

        st.dataframe(
            cluster_df.head(100),
            width="stretch"
        )


# =========================================================
# 12. WEBSITE FOOTER
# =========================================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    padding:14px 5px 4px;
    color:#8795ad;
    font-size:11px;
    line-height:2;
">
    <span style="font-weight:800;color:#1769ff">RideRush</span>
    &nbsp; · &nbsp; AI-Based Trip Demand Forecasting and ETA Prediction
    <br>
    Machine Learning Project &nbsp; | &nbsp; NYC Yellow Taxi Dataset
</div>
""", unsafe_allow_html=True)
