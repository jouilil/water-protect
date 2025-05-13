import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Set Streamlit page configuration (must be first)
st.set_page_config(layout="wide", page_title="Global Water Sales Dashboard")

# ✅ Custom Styling for Dark Theme and Bold White Title
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .block-container {
            background-color: #121212;
        }
        .stSelectbox, .stMultiSelect {
            background-color: #333333;
            color: white;
            border: 1px solid #444;
        }
        .stButton {
            background-color: #1976d2;
            color: white;
        }
        .stTextInput, .stTextArea {
            background-color: #333333;
            color: white;
            border: 1px solid #444;
        }
        .stSlider, .stRadio {
            background-color: #333333;
            color: white;
            border: 1px solid #444;
        }
        .stMarkdown {
            color: white;
        }
        .css-1y5i3j3 {  /* Sidebar specific styling */
            background-color: #121212;
        }
        .stTitle {
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/cc1.csv")
    df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')  # Extract valid 4-digit years
    df = df.dropna(subset=['year'])  # Drop rows where year couldn't be extracted
    df['year'] = df['year'].astype(int)
    
    # Clean the 'Consumption' column to remove spaces and convert to float
    df['Consumption'] = df['Consumption'].astype(str).str.replace(' ', '').astype(float)

    return df

df = load_data()

# ✅ Forecast function with multiple ML models
@st.cache_data
def forecast_sales(df, operator):
    operator_data = df[df['OPERATEUR'] == operator].copy()
    operator_data = operator_data.sort_values('year')

    X = operator_data[['year']]
    y = operator_data['Consumption']

    # Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Support Vector Regression": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    }

    future_years = np.arange(2020, 2027)
    forecast_data = {}

    for model_name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(future_years.reshape(-1, 1))
        residuals = y - model.predict(X)
        std_dev = np.std(residuals)
        ci = 1.96 * std_dev

        forecast_data[model_name] = pd.DataFrame({
            "year": future_years,
            "prediction": y_pred,
            "lower": y_pred - ci,
            "upper": y_pred + ci
        })

    return operator_data, forecast_data

# ✅ Dashboard Header with Bold and White Title
col1, col2 = st.columns([1, 3])
with col1:
    st.image("logo.JPG", width=130)
with col2:
    st.markdown("""
        <h1 style="color: blue; font-weight: bold;">💧 Global Water Sales Dashboard</h1>
    """, unsafe_allow_html=True)

st.markdown(
    """
    This interactive dashboard provides a comprehensive overview of water sales patterns
    by operator from 2020 to 2024 (from January to August). It is designed to support data-driven decision-making through three
    main visual components:

    1. A multi-operator time series chart illustrating historical trends;

    2. A dynamic pie chart showing the annual distribution of sales for a selected operator;

    3. A predictive analytics section projecting future water sales using multiple machine learning models
    including Linear Regression, Random Forest, Decision Tree, and SVM models, with confidence intervals.

    The dashboard enables users to explore past performance, assess current usage, and anticipate future needs
    within a unified analytical framework.
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ✅ Sidebar for Controls
with st.sidebar:
    st.header("⚙️ Controls")
    selected_operator = st.selectbox("Select an Operator:", sorted(df["OPERATEUR"].unique()))
    selected_models = st.multiselect(
        "Select Forecast Models:",
        ["Linear Regression", "Random Forest", "Decision Tree", "Support Vector Regression"],
        default=["Linear Regression", "Random Forest"]
    )

# ✅ Main Area for Charts
st.markdown("<h3 style='color:white; font-weight:bold;'>📊 Data Visualizations</h3>", unsafe_allow_html=True)


# ✅ Nouveau graphique radar : Consommation par année (2020–2024 forcées)
st.markdown("<h3 style='color:white; font-weight:bold;'>🔍 Annual Sales Comparison</h3>", unsafe_allow_html=True)

# Filtrer les données pour l'opérateur sélectionné
filtered_radar = df[df["OPERATEUR"] == selected_operator]

# Forcer les années 2020 à 2024
all_years = pd.DataFrame({"year": [2020, 2021, 2022, 2023, 2024]})
yearly_consumption = filtered_radar.groupby("year")["Consumption"].sum().reset_index()
yearly_consumption = pd.merge(all_years, yearly_consumption, on="year", how="left").fillna(0)

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=yearly_consumption["Consumption"],
    theta=yearly_consumption["year"].astype(str),
    fill='toself',
    name=selected_operator,
    line=dict(color='deepskyblue')
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, yearly_consumption["Consumption"].max() * 1.1]
        ),
        angularaxis=dict(
            direction='clockwise',
            rotation=90
        )
    ),
    showlegend=False,
    title=f"Radar de la Consommation d'Eau (2020–2024) - {selected_operator}",
    paper_bgcolor="#121212",
    font_color="white"
)

st.plotly_chart(fig_radar, use_container_width=True)

# First Chart: Line Chart with its title
st.markdown("<h3 style='color:white; font-weight:bold;'>📉 Annual Water Sales by Operator</h3>", unsafe_allow_html=True)
line_fig = px.line(
    df,
    x="year",
    y="Consumption",
    color="OPERATEUR",
    markers=True,
    title="Annual Sales by Operator",
    labels={"Consumption": "Water Sales (m³)", "year": "Year", "OPERATEUR": "Operator"}
).update_layout(
    paper_bgcolor="#121212",
    plot_bgcolor="#121212",
    font_color="white",
    xaxis=dict(tickmode='linear', tickformat='d'),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5,
        font=dict(size=12, color='white'),
        bgcolor='rgba(0, 0, 0, 0.5)',
        borderwidth=0
    )
)
st.plotly_chart(line_fig, use_container_width=True)

# Second Chart: Pie Chart in a new section
st.markdown(f"<h3 style='color:white; font-weight:bold;'>⭕ Yearly Consumption Share for {selected_operator}</h3>", unsafe_allow_html=True)

filtered_pie = df[df["OPERATEUR"] == selected_operator]
pie_fig = px.pie(
    filtered_pie,
    names="year",
    values="Consumption",
    title=f"Yearly Consumption Share for {selected_operator}",
    hole=0.3,
    labels={"year": "Year", "Consumption": "Sales (m³)"}
).update_layout(
    paper_bgcolor="#121212",
    plot_bgcolor="#121212",
    font_color="white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5,
        font=dict(size=12, color='white'),
        bgcolor='rgba(0, 0, 0, 0.5)',
        borderwidth=0
    )
)
st.plotly_chart(pie_fig, use_container_width=True)

# Forecast Chart
st.markdown(f"<h3 style='color:white; font-weight:bold;'>🔮 Forecasted Water Sales for {selected_operator} (2020–2026)</h3>", unsafe_allow_html=True)
actual_data, forecast_data = forecast_sales(df, selected_operator)
forecast_fig = go.Figure()

forecast_fig.add_trace(go.Scatter(
    x=actual_data['year'],
    y=actual_data['Consumption'],
    mode='lines+markers',
    name='Actual',
    line=dict(color='cyan')
))

model_colors = {
    "Linear Regression": '#FFA500',
    "Random Forest": '#228B22',
    "Decision Tree": '#1E90FF',
    "Support Vector Regression": '#800080'
}

for model_name in selected_models:
    if model_name in forecast_data:
        forecast = forecast_data[model_name]
        color = model_colors[model_name]
        rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        forecast_fig.add_trace(go.Scatter(
            x=forecast['year'],
            y=forecast['prediction'],
            mode='lines+markers',
            name=f'{model_name} Forecast',
            line=dict(color=color, dash='dash')
        ))

        forecast_fig.add_trace(go.Scatter(
            x=list(forecast['year']) + list(forecast['year'][::-1]),
            y=list(forecast['upper']) + list(forecast['lower'][::-1]),
            fill='toself',
            fillcolor=f'rgba{rgb + (0.2,)}',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name=f'{model_name} 95% CI'
        ))

forecast_fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Sales (m³)",
    paper_bgcolor="#121212",
    plot_bgcolor="#121212",
    font_color="white",
    legend=dict(
        orientation="h", 
        yanchor="bottom", 
        y=-0.4, 
        xanchor="center", 
        x=0.5,
        font=dict(size=12, color='white'),
        bgcolor='rgba(0, 0, 0, 0.5)',
        borderwidth=0
    )
)
st.plotly_chart(forecast_fig, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: lightgray; font-style: italic; font-size: 14px;'>© May, 2025 | Dashboard Developed by Mr. Bougantouche & Mr. Bouceta</p>",
    unsafe_allow_html=True
)
