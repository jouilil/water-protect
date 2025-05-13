import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# ✅ Configuration de la page Streamlit (doit être placée en premier)
st.set_page_config(layout="wide", page_title="Tableau de Bord Global des Ventes d'Eau")

# ✅ Style personnalisé pour thème sombre et titre en blanc gras
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
        .css-1y5i3j3 {
            background-color: #121212;
        }
        .stTitle {
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv("data/cc1.csv")
    df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')  # Extraire les années valides
    df = df.dropna(subset=['year'])  # Supprimer les lignes sans année valide
    df['year'] = df['year'].astype(int)

    # Nettoyer la colonne 'Consumption' pour supprimer les espaces et convertir en float
    df['Consumption'] = df['Consumption'].astype(str).str.replace(' ', '').astype(float)

    return df

df = load_data()

# ✅ Fonction de prévision avec plusieurs modèles ML
@st.cache_data
def forecast_sales(df, operator):
    operator_data = df[df['OPERATEUR'] == operator].copy()
    operator_data = operator_data.sort_values('year')

    X = operator_data[['year']]
    y = operator_data['Consumption']

    # Modèles
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

# ✅ En-tête du tableau de bord
col1, col2 = st.columns([1, 3])
with col1:
    st.image("logo.JPG", width=130)
with col2:
    st.markdown("""
        <h1 style="color: blue; font-weight: bold;">💧 Tableau de Bord Global des Ventes d'Eau</h1>
    """, unsafe_allow_html=True)

st.markdown(
    """
    Ce tableau de bord interactif fournit une vue d'ensemble complète des ventes d'eau
    par opérateur de 2020 à 2024 (de janvier à août). Il est conçu pour faciliter la prise de décision fondée sur les données à travers trois
    composants visuels principaux :

    1. Une série chronologique multi-opérateurs montrant les tendances historiques ;

    2. Un graphique circulaire dynamique présentant la répartition annuelle des ventes pour un opérateur sélectionné ;

    3. Une section de prévisions utilisant plusieurs modèles d'apprentissage automatique
    incluant la régression linéaire, la forêt aléatoire, l'arbre de décision et la SVM, avec des intervalles de confiance.

    Le tableau de bord permet aux utilisateurs d'explorer les performances passées, d’évaluer la consommation actuelle
    et d’anticiper les besoins futurs dans un cadre analytique unifié.
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ✅ Barre latérale de contrôle
with st.sidebar:
    st.header("⚙️ Contrôles")
    selected_operator = st.selectbox("Sélectionnez un opérateur :", sorted(df["OPERATEUR"].unique()))
    selected_models = st.multiselect(
        "Sélectionnez les modèles de prévision :",
        ["Régression Linéaire", "Forêt Aléatoire", "Arbre de Décision", "Régression à Vecteurs de Support"],
        default=["Régression Linéaire", "Forêt Aléatoire"]
    )

# ✅ Zone principale pour les graphiques
st.markdown("<h3 style='color:white; font-weight:bold;'>📊 Visualisations des Données</h3>", unsafe_allow_html=True)

# ✅ Graphique radar annuel
st.markdown("<h3 style='color:white; font-weight:bold;'>🔍 Comparaison Annuelle des Ventes</h3>", unsafe_allow_html=True)

filtered_radar = df[df["OPERATEUR"] == selected_operator]

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

# ✅ Graphique en ligne
st.markdown("<h3 style='color:white; font-weight:bold;'>📉 Ventes Annuelles d'Eau par Opérateur</h3>", unsafe_allow_html=True)
line_fig = px.line(
    df,
    x="year",
    y="Consumption",
    color="OPERATEUR",
    markers=True,
    title="Ventes Annuelles par Opérateur",
    labels={"Consumption": "Ventes d'eau (m³)", "year": "Année", "OPERATEUR": "Opérateur"}
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

# ✅ Graphique circulaire
st.markdown(f"<h3 style='color:white; font-weight:bold;'>⭕ Part Annuelle de la Consommation - {selected_operator}</h3>", unsafe_allow_html=True)

filtered_pie = df[df["OPERATEUR"] == selected_operator]
pie_fig = px.pie(
    filtered_pie,
    names="year",
    values="Consumption",
    title=f"Répartition Annuelle de la Consommation - {selected_operator}",
    hole=0.3,
    labels={"year": "Année", "Consumption": "Ventes (m³)"}
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

# ✅ Graphique de prévision
st.markdown(f"<h3 style='color:white; font-weight:bold;'>🔮 Prévision des Ventes d'Eau pour {selected_operator} (2020–2026)</h3>", unsafe_allow_html=True)
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
    yaxis_title="Ventes (m³)",
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

# ✅ Pied de page
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: lightgray; font-style: italic; font-size: 14px;'>© Mai 2025 | Tableau de bord développé par M. Bougantouche & M. Bouceta</p>",
    unsafe_allow_html=True
)
