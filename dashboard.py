import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="RideRush | ML Dashboard",
    page_icon="🚕",
    layout="wide"
)

# Blue theme and styling
st.markdown("""
<style>
.stApp {
    background-color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background-color: #0f172a;
}
[data-testid="stSidebar"] * {
    color: white;
}
h1, h2, h3 {
    color: #1e3a8a;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dbeafe;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🚕 RideRush")
st.subheader("AI-Based Trip Demand Forecasting and ETA Prediction")
st.caption("Machine Learning | Demand Forecasting | Trip Duration Prediction")

# Navigation
st.sidebar.title("🚕 RideRush")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Dataset Analysis",
        "Demand Analysis",
        "ETA Prediction",
        "Model Evaluation"
    ]
)

st.sidebar.divider()
st.sidebar.info("AI-powered trip analytics dashboard")

# Sample data for visualization
hours = list(range(24))
demand = [
    20, 12, 8, 6, 5, 10, 25, 45,
    65, 80, 60, 50, 55, 58, 62, 70,
    85, 95, 90, 75, 60, 50, 40, 30
]

# OVERVIEW
if page == "Overview":

    st.header("Dashboard Overview")
    st.write("Monitor trip demand and explore estimated trip durations.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Project", "RideRush")
    c2.metric("ML Tasks", "2")
    c3.metric("Peak Demand Hour", "5 PM")

    st.markdown("### 📊 Hourly Trip Demand")

    chart_df = pd.DataFrame({
        "Hour": hours,
        "Demand": demand
    })

    fig = px.area(
        chart_df,
        x="Hour",
        y="Demand",
        markers=True,
        title="Illustrative Demand Throughout the Day"
    )

    fig.update_traces(line_color="#2563eb")
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Hour of Day",
        yaxis_title="Relative Demand"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🚀 Project Objectives")
    c1, c2 = st.columns(2)

    with c1:
        st.info("**Demand Forecasting**\n\nIdentify hours with higher trip demand.")

    with c2:
        st.info("**ETA Prediction**\n\nEstimate how long a trip may take.")

# DATASET ANALYSIS
elif page == "Dataset Analysis":

    st.header("📁 Dataset Analysis")
    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.markdown("### Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.markdown("### Missing Values by Column")
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if not missing.empty:
            st.bar_chart(missing)
        else:
            st.success("No missing values found.")

    else:
        st.info("Upload your project CSV file to explore the dataset.")

# DEMAND ANALYSIS
elif page == "Demand Analysis":

    st.header("📈 Demand Analysis")
    st.write("Explore the illustrative hourly demand pattern.")

    selected_hour = st.slider("Select Hour", 0, 23, 17)

    chart_df = pd.DataFrame({
        "Hour": hours,
        "Demand": demand
    })

    fig = px.bar(
        chart_df,
        x="Hour",
        y="Demand",
        color="Demand",
        color_continuous_scale="Blues",
        title="Hourly Demand Pattern"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.metric("Selected Hour Demand", demand[selected_hour])

    if selected_hour in [7, 8, 9, 16, 17, 18, 19]:
        st.warning("This hour is within the project's defined rush-hour period.")
    else:
        st.success("This hour is outside the project's defined rush-hour period.")

# ETA PREDICTION
elif page == "ETA Prediction":

    st.header("⏱️ Trip Duration Estimator")
    st.write("Enter trip details to see a demonstration estimate.")

    c1, c2 = st.columns(2)

    with c1:
        distance = st.number_input(
            "Trip Distance (miles)",
            min_value=0.1,
            max_value=100.0,
            value=5.0
        )

    with c2:
        passengers = st.number_input(
            "Passenger Count",
            min_value=1,
            max_value=8,
            value=1
        )

    pickup_hour = st.slider("Pickup Hour", 0, 23, 12)

    if st.button("Estimate Trip Duration", type="primary"):
        duration = distance * 3 + 5

        if pickup_hour in [7, 8, 9, 16, 17, 18, 19]:
            duration *= 1.3

        st.markdown("### Prediction Result")
        c1, c2 = st.columns(2)
        c1.metric("Estimated Duration", f"{duration:.1f} minutes")
        c2.metric("Distance", f"{distance:.1f} miles")

        st.caption(
            "Demonstration estimate only; this does not use the trained ML model."
        )

# MODEL EVALUATION
elif page == "Model Evaluation":

    st.header("🎯 Model Evaluation")
    st.write("Example metrics from the RideRush Random Forest experiment.")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", "5.22 min")
    c2.metric("RMSE", "8.61 min")
    c3.metric("R² Score", "0.75")

    st.markdown("### What These Metrics Mean")

    st.write("- **MAE:** Average absolute prediction error.")
    st.write("- **RMSE:** Gives greater weight to large prediction errors.")
    st.write("- **R² Score:** Measures how much variation the model explains.")

    st.caption(
        "These are example project results, not calculated during dashboard execution."
    )