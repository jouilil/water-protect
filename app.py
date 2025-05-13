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
        .stSelectbox, .stMultiSelect {
            background-color: #f0f0f0;
            color: black;
            border: 1px solid #ccc;
        }
        .stButton {
            background-color: #1976d2;
            color: white;
        }
        .stTextInput, .stTextArea {
            background-color: #f0f0f0;
            color: black;
            border: 1px solid #ccc;
        }
        .stSlider, .stRadio {
            background-color: #f0f0f0;
            color: black;
            border: 1px solid #ccc;
        }
        .css-1y5i3j3 {
            background-color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv("data/cc1.csv")
    df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    df['Consumption'] = df['Consumption'].astype(str).str.replace(' ', '').astype(float)
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
    Ce tableau de bord interactif fournit une vue d'ensemble complète des ventes d'eau
    par opérateur de 2020 à 2024 (janvier à août). Il permet :

    1. Une visualisation des tendances historiques par opérateur ;
    2. Une répartition annuelle des ventes sous forme de graphique circulaire ;
    3. Une prévision basée sur plusieurs modèles de Machine Learning.
""", unsafe_allow_html=True)

# ✅ Barre latérale
with st.sidebar:
    st.header("⚙️ Contrôles")
    selected_operator = st.selectbox("Sélectionnez un opérateur :", sorted(df["OPERATEUR"].unique()))
    selected_models = st.multiselect(
        "Sélectionnez les modèles de prévision :",
        ["Régression Linéaire", "Forêt Aléatoire", "Arbre de Décision", "Régression à Vecteurs de Support"],
        default=["Régression Linéaire", "Forêt Aléatoire"]
    )

# ✅ Titre section visualisation
st.markdown("<h3>📊 Visualisations des Données</h3>", unsafe_allow_html=True)

# ✅ Radar
st.markdown("<h3>🔍 Radar Interactif : Comparaison des Opérateurs pour les Années Sélectionnées</h3>", unsafe_allow_html=True)

# Choix des années via un multiselect
available_years = [2020, 2021, 2022, 2023, 2024]
selected_years = st.multiselect(
    "Sélectionnez une ou plusieurs années :", 
    options=available_years, 
    default=[2020]  # Valeur par défaut
)

# Vérifier si au moins une année est sélectionnée
if selected_years:
    # Filtrer les données pour les années sélectionnées
    filtered_data = df[df["year"].isin(selected_years)]
    
    # Liste des opérateurs à afficher sur les axes du radar
    operators = filtered_data["OPERATEUR"].unique()

    # Préparer les données pour le radar
    fig_radar = go.Figure()

    # Ajouter une ligne pour chaque opérateur, en combinant les années sélectionnées
    fig_radar.add_trace(go.Scatterpolar(
        r=[filtered_data[filtered_data["OPERATEUR"] == op]["Consumption"].sum() for op in operators],
        theta=operators,  # Les opérateurs comme axes angulaires
        fill='toself',
        name=f"Consommation - {', '.join(map(str, selected_years))}",
        line=dict(color='deepskyblue')
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, filtered_data["Consumption"].max() * 1.1]),
            angularaxis=dict(direction='clockwise', rotation=90)  # Pour que le premier opérateur soit en haut
        ),
        title=f"Radar de Consommation d'Eau par Opérateur pour les Années {', '.join(map(str, selected_years))}",
        showlegend=True,
        paper_bgcolor="white",
        font_color="black"
    )

    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.warning("Veuillez sélectionner au moins une année.")

# ✅ Ligne
st.markdown("<h3>📉 Ventes Annuelles d'Eau par Opérateur</h3>", unsafe_allow_html=True)
line_fig = px.line(
    df,
    x="year",
    y="Consumption",
    color="OPERATEUR",
    markers=True,
    labels={"Consumption": "Ventes d'eau (m³)", "year": "Année", "OPERATEUR": "Opérateur"},
    color_discrete_sequence=px.colors.qualitative.Set2  # Palette harmonieuse
)

line_fig.update_traces(line=dict(width=3), marker=dict(size=6, symbol="circle"))

line_fig.update_layout(
    title_text="Évolution des ventes annuelles d'eau par opérateur (2020–2024)",
    title_font=dict(size=18, color='black'),
    xaxis=dict(
        tickmode='linear',
        tickformat='d',
        title='Année',
        title_font=dict(size=14),
        tickfont=dict(size=12)
    ),
    yaxis=dict(
        title='Ventes d\'eau (m³)',
        title_font=dict(size=14),
        tickfont=dict(size=12)
    ),
    legend=dict(
        title='',
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=12),
        bgcolor='rgba(0,0,0,0)'
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="black")
)

st.plotly_chart(line_fig, use_container_width=True)

# ✅ Camembert
st.markdown(f"<h3>⭕ Part Annuelle de la Consommation - {selected_operator}</h3>", unsafe_allow_html=True)
filtered_pie = df[df["OPERATEUR"] == selected_operator]
pie_fig = px.pie(
    filtered_pie,
    names="year",
    values="Consumption",
    title=f"Répartition Annuelle de la Consommation - {selected_operator}",
    hole=0.3,
    labels={"year": "Année", "Consumption": "Ventes (m³)"}
).update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font_color="black",
    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5,
                font=dict(size=12, color='black'),
                bgcolor='rgba(0, 0, 0, 0.05)', borderwidth=0)
)
st.plotly_chart(pie_fig, use_container_width=True)

# ✅ Prévision
st.markdown(f"<h3>🔮 Prévision des Ventes d'Eau pour {selected_operator} (2020–2026)</h3>", unsafe_allow_html=True)
actual_data, forecast_data = forecast_sales(df, selected_operator)
forecast_fig = go.Figure()

forecast_fig.add_trace(go.Scatter(
    x=actual_data['year'],
    y=actual_data['Consumption'],
    mode='lines+markers',
    name='Réel',
    line=dict(color='cyan')
))

model_colors = {
    "Régression Linéaire": '#FFA500',
    "Forêt Aléatoire": '#228B22',
    "Arbre de Décision": '#1E90FF',
    "Régression à Vecteurs de Support": '#800080'
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
            name=f'Prévision {model_name}',
            line=dict(color=color, dash='dash')
        ))
        forecast_fig.add_trace(go.Scatter(
            x=list(forecast['year']) + list(forecast['year'][::-1]),
            y=list(forecast['upper']) + list(forecast['lower'][::-1]),
            fill='toself',
            fillcolor=f'rgba{rgb + (0.2,)}',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name=f'{model_name} IC à 95%'
        ))

forecast_fig.update_layout(
    xaxis_title="Année",
    yaxis_title="Ventes",
    paper_bgcolor="white",
    plot_bgcolor="white",
    font_color="black",
    legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5,
                font=dict(size=12, color='black'),
                bgcolor='rgba(0, 0, 0, 0.05)', borderwidth=0)
)
st.plotly_chart(forecast_fig, use_container_width=True)

# ✅ Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray; font-style: italic; font-size: 14px;'>© Mai 2025 | Tableau de bord développé par M. Bougantouche & M. Bouceta</p>",
    unsafe_allow_html=True
)
