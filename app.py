import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# ✅ Configuration de la page Streamlit
st.set_page_config(layout="wide", page_title="Global Water Sales Dashboard")

# ✅ Style personnalisé
st.markdown("""
    <style>
        body {
            background-color: white;
            color: black;
        }
        .stTitle, .stHeader, .stMarkdown h3, h3, h1 {
            color: black !important;
            font-weight: bold !important;
        }
        .block-container {
            background-color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv("data/cc1.csv")
    return df

df = load_data()

# ✅ Fonction de prévision
@st.cache_data
def forecast_sales(df, operator):
    operator_data = df[df['OPERATEUR'] == operator].copy()
    operator_data = operator_data.sort_values('year')

    X = operator_data[['year']]
    y = operator_data['Consumption']

    models = {
        "Régression Linéaire": LinearRegression(),
        "Forêt Aléatoire": RandomForestRegressor(n_estimators=100, random_state=42),
        "Arbre de Décision": DecisionTreeRegressor(random_state=42),
        "Régression à Vecteurs de Support": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
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

# ✅ En-tête
col1, col2 = st.columns([1, 3])
with col1:
    st.image("logo.JPG", width=130)
with col2:
    st.markdown("<h1 style='color:blue; font-weight:bold;'>💧 Global Water Sales Dashboard</h1>", unsafe_allow_html=True)

st.markdown("""
    Ce tableau de bord interactif fournit une vue d'ensemble des ventes d'eau
    par opérateur de 2020 à 2024 (janvier à août).
""", unsafe_allow_html=True)

# ✅ Barre latérale
with st.sidebar:
    st.header("⚙️ Contrôles")
    selected_operator = st.selectbox("Sélectionnez un opérateur :", sorted(df["OPERATEUR"].unique()))
    selected_models = st.multiselect(
        "Sélectionnez les modèles de prévision :",
        ["Régression Linéaire", "Forêt Aléatoire", "Arbre de Décision", "Régression à Vecteurs de Support"],
        default=["Régression Linéaire"]
    )

# ✅ RADAR avec checkboxes pour les années
st.markdown("<h3>🔍 Radar : Comparaison des Ventes par Année</h3>", unsafe_allow_html=True)

# Détection des années disponibles dans les données
available_years = sorted(df["year"].unique())

# Checkboxes pour chaque année
years_selected = [year for year in available_years if st.checkbox(f"Afficher {year}", value=True)]

# Filtrer les données pour le radar
filtered_radar = df[(df["OPERATEUR"] == selected_operator) & (df["year"].isin(years_selected))]

# Si aucune année sélectionnée, ne pas afficher de radar
if years_selected and not filtered_radar.empty:
    grouped_radar = filtered_radar.groupby("year")["Consumption"].sum().reset_index()
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=grouped_radar["Consumption"],
        theta=grouped_radar["year"].astype(str),
        fill='toself',
        name=selected_operator
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True),
            angularaxis=dict(direction='clockwise')
        ),
        showlegend=False,
        title=f"Consommation annuelle pour {selected_operator}"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.warning("Veuillez sélectionner au moins une année pour afficher le radar.")

# ✅ LIGNE : évolution annuelle
st.markdown("<h3>📈 Évolution des Ventes d’Eau par Opérateur</h3>", unsafe_allow_html=True)
line_fig = px.line(
    df,
    x="year",
    y="Consumption",
    color="OPERATEUR",
    markers=True
)
st.plotly_chart(line_fig, use_container_width=True)

# ✅ CAMEMBERT : répartition annuelle
st.markdown(f"<h3>🍰 Répartition Annuelle de la Consommation – {selected_operator}</h3>", unsafe_allow_html=True)
filtered_pie = df[df["OPERATEUR"] == selected_operator]
pie_fig = px.pie(
    filtered_pie,
    names="year",
    values="Consumption",
    hole=0.3
)
st.plotly_chart(pie_fig, use_container_width=True)

# ✅ PRÉVISIONS
st.markdown(f"<h3>🔮 Prévisions des Ventes pour {selected_operator} (2020–2026)</h3>", unsafe_allow_html=True)
actual_data, forecast_data = forecast_sales(df, selected_operator)
forecast_fig = go.Figure()

# Réel
forecast_fig.add_trace(go.Scatter(
    x=actual_data['year'],
    y=actual_data['Consumption'],
    mode='lines+markers',
    name='Historique'
))

# Modèles
for model_name in selected_models:
    forecast = forecast_data[model_name]
    forecast_fig.add_trace(go.Scatter(
        x=forecast["year"],
        y=forecast["prediction"],
        mode='lines+markers',
        name=f"{model_name}"
    ))

forecast_fig.update_layout(
    xaxis_title="Année",
    yaxis_title="Ventes d'eau",
    legend_title="Modèle"
)
st.plotly_chart(forecast_fig, use_container_width=True)
