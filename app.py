import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from datetime import datetime

# ✅ Configuration de la page Streamlit
st.set_page_config(layout="wide", page_title= "Evaluation globale de la consommation d'eau potable domestique par les différents usages")

# ✅ Barre latérale de navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choisir une interface :", ["Données officielles", "Enquête Terrain"])

if page == "Données officielles":
    # 🔽 VOTRE CODE ACTUEL ICI 🔽
    # load_data, forecast_sales, st.image, st.markdown, st.plotly_chart, etc.
    # ... (tout ce que vous avez déjà fourni)

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
        st.markdown("<h1 style='color:blue; font-weight:bold;'>💧 Global Water Consumption Dashboard</h1>", unsafe_allow_html=True)

    # Date et heure actuelles en français
    current_datetime = datetime.now().strftime("%d %B %Y %H:%M:%S")

    # Affichage dans Streamlit
    st.markdown(f"<p><strong>Dernière mise à jour :</strong> {current_datetime}</p>", unsafe_allow_html=True)


    st.markdown("""

    <h2> <strong> 1. Introduction </strong></h2>

    <p> Ce Dashboard 💧 interactif offre une vue analytique complète de la consommation d’eau par opérateur au sein des ports marocains, couvrant la période allant de 2020 à 2024 (janvier à août). Développé pour faciliter la compréhension des dynamiques de consommation et appuyer la prise de décision stratégique, il intègre plusieurs modules de visualisation et d’analyse prédictive.</p>

    <p>Ce Dashboard se structure en quatre volets principaux :</p>

    <ol>
    <li><strong>Analyse historique de la consommation par opérateur</strong> : un graphique linéaire interactif permet de visualiser l’évolution annuelle des volumes d’eau vendus par chaque opérateur, mettant en évidence les tendances, pics et éventuelles ruptures.</li>
    <li><strong>Répartition annuelle sous forme de graphique circulaire</strong> : cette visualisation met en relief la part relative de chaque année dans la consommation globale d’un opérateur donné, facilitant la comparaison entre exercices.</li>
    <li><strong>Comparaison inter-opérateurs par radar</strong> : une représentation radiale permet de comparer visuellement les volumes annuels de consommation entre opérateurs, avec la possibilité de sélectionner dynamiquement les années à analyser.</li>
    <li><strong>Module de prévision par Machine Learning</strong> : basé sur plusieurs algorithmes (régression linéaire, forêt aléatoire, arbre de décision et SVM), ce module propose des projections de la demande future, assorties d’intervalles de confiance pour mieux anticiper les évolutions.</li>
    </ol>

    <p>Grâce à une interface épurée, des filtres dynamiques et des représentations graphiques adaptées, ce tableau de bord constitue un outil décisionnel robuste pour les gestionnaires, les analystes et les acteurs institutionnels impliqués dans la gestion durable des ressources hydriques.</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <hr>
    <h3>⚙️ <strong>Technologies et outils utilisés</strong></h3>

    <p>Le développement de ce Dashboard interactif repose sur une combinaison d'outils et de technologies permettant de garantir à la fois flexibilité, performance et interactivité.</p>

    <ul>
        <li><strong>Python</strong> : Le langage de programmation principal utilisé pour la manipulation des données, le calcul des prévisions, et l'intégration des modèles de Machine Learning.</li>
        <li><strong>Streamlit</strong> : Un framework Python permettant de développer des applications web interactives. Il a été choisi pour sa simplicité d’utilisation et sa capacité à générer rapidement des interfaces utilisateur performantes et esthétiques.</li>
        <li><strong>Pandas</strong> : Une bibliothèque Python pour la gestion et la manipulation de données structurées. Elle est utilisée pour le prétraitement et l'agrégation des données historiques sur la consommation d’eau.</li>
        <li><strong>Plotly</strong> : Utilisée pour créer des visualisations interactives. Plotly permet de générer des graphiques dynamiques et des cartes, adaptés aux besoins de visualisation des tendances de la consommation et des prévisions.</li>
        <li><strong>Scikit-learn</strong> : Bibliothèque spécialisée dans le Machine Learning, utilisée pour les modèles de régression et de prévision, notamment la régression linéaire, l'arbre de décision et la forêt aléatoire.</li>
        <li><strong>Visual Studio Code (VS Code)</strong> : L'environnement de développement intégré (IDE) choisi pour la rédaction du code, permettant une gestion claire du projet grâce à ses fonctionnalités de débogage, de gestion de versions et d'intégration d'extensions Python.</li>
    </ul>

    <p>En combinant ces outils, le tableau de bord offre une solution robuste et évolutive pour l'analyse de la consommation d'eau, capable de répondre à différents besoins d'analyse et de décision, tout en restant facile à utiliser pour les utilisateurs finaux.</p>
    """, unsafe_allow_html=True)

    # ✅ Barre latérale

    # ✅ Titre section visualisation
    st.markdown("<h2>📊 2. Description et Visualisations des Données</h2>", unsafe_allow_html=True)

    # ✅ Ligne
    # ✅ Graphique des ventes annuelles de tous les opérateurs
    st.markdown("<h3>📈 2.1 Consommation d'Eau au fil du temps- Tous les Opérateurs</h3>", unsafe_allow_html=True)

    # Regrouper les données par opérateur et année
    grouped_all = df.groupby(['year', 'OPERATEUR'])['Consumption'].sum().reset_index()

    # Créer la figure
    fig_all_operators = go.Figure()

    # Tracer une ligne par opérateur
    for operator in grouped_all['OPERATEUR'].unique():
        operator_data = grouped_all[grouped_all['OPERATEUR'] == operator]
        fig_all_operators.add_trace(go.Scatter(
            x=operator_data['year'],
            y=operator_data['Consumption'],
            mode='lines+markers',
            name=operator
        ))

    # Mise en forme
    fig_all_operators.update_layout(
        xaxis_title="Année",
        yaxis_title="Consommation ",
        title="",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="black",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        )
    )

    st.plotly_chart(fig_all_operators, use_container_width=True)
    # ✅ Radar
    # ✅ Radar avec les opérateurs comme angles et les années sélectionnables
    # ✅ Radar avec couleurs différentes pour chaque année sélectionnée
    st.markdown("<h3>🔍 2.2 Comparaison Annuelle par Opérateur</h3>", unsafe_allow_html=True)

    # Liste des années disponibles
    available_years = sorted(df["year"].unique())

    # Checkboxes pour sélectionner les années à comparer
    st.markdown("**Sélectionnez les années à comparer :**")
    selected_years = []
    cols = st.columns(len(available_years))
    for i, year in enumerate(available_years):
        if cols[i].checkbox(str(year), value=(year == max(available_years))):  # coche la plus récente par défaut
            selected_years.append(year)

    # Si aucune année sélectionnée, afficher un avertissement
    if not selected_years:
        st.warning("Veuillez sélectionner au moins une année pour afficher le radar.")
    else:
        # Palette de couleurs pour distinguer les années
        color_palette = px.colors.qualitative.Set2 + px.colors.qualitative.Plotly

        # Créer le radar chart
        fig_radar = go.Figure()

        for i, year in enumerate(selected_years):
            yearly_data = df[df["year"] == year]
            radar_summary = yearly_data.groupby("OPERATEUR")["Consumption"].sum().reset_index()

            fig_radar.add_trace(go.Scatterpolar(
                r=radar_summary["Consumption"],
                theta=radar_summary["OPERATEUR"],
                fill='toself',
                name=str(year),
                line=dict(color=color_palette[i % len(color_palette)])
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, df[df['year'].isin(selected_years)]['Consumption'].max() * 1.1])
            ),
            showlegend=True,
            title=f"Radar de consommation par Opérateur ({', '.join(map(str, selected_years))})",
            paper_bgcolor="white",
            font_color="black",
            legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
        )

        st.plotly_chart(fig_radar, use_container_width=True)



    # ✅ Camembert

    # Contrôles de sélection dans la section 2.3
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>⚙️ Analyse et Prévision future </h3>", unsafe_allow_html=True)
    st.write("L’objectif de cette section est de permettre à l’utilisateur de sélectionner un opérateur afin d’accéder aux modèles de prévision et aux statistiques annuelles, tout en organisant l’interface de manière claire et structurée grâce à une disposition en colonnes. Merci de bien vouloir choisir un opérateur")
    st.write("Merci de sélectionner un opérateur")

    col1, col2 = st.columns([2, 3])
    with col1:
        selected_operator = st.selectbox("Sélectionnez un opérateur :", sorted(df["OPERATEUR"].unique()), key="operator_select_2_3")
    with col2:
        selected_models = st.multiselect(
            "Sélectionnez les modèles de prévision :",
            ["Régression Linéaire", "Forêt Aléatoire", "Arbre de Décision", "Régression à Vecteurs de Support"],
            default=["Régression Linéaire", "Forêt Aléatoire"],
            key="model_select_2_3"
        )

    st.markdown(f"<h3>⭕ 2.3  Part Annuelle de la Consommation - {selected_operator}</h3>", unsafe_allow_html=True)
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
    st.markdown(f"<h3>3. 🔮 Prévision de la consommation d'Eau pour {selected_operator} (2020–2026)</h3>", unsafe_allow_html=True)
    st.markdown("""

        <p> Cette section  se concentre sur l'estimation de cla onsommation future d'eau à travers différents modèles de Machine Learning. En exploitant des méthodes statistiques avancées et des algorithmes d'apprentissage automatique, nous fournissons des prévisions basées sur les données historiques des opérateurs.</p>

        <p>Les modèles de prévision utilisés sont :</p>
        <ul>
            <li><strong>Régression Linéaire</strong> : Ce modèle simple mais puissant est utilisé pour établir une relation linéaire entre l'année et la consommation d'eau, permettant ainsi de prévoir les tendances futures.</li>
            <li><strong>Forêt Aléatoire</strong> : Un modèle d'ensemble qui construit plusieurs arbres de décision pour améliorer la précision des prévisions, tout en réduisant le risque de surajustement (overfitting).</li>
            <li><strong>Arbre de Décision</strong> : Un modèle non linéaire qui divise les données en sous-groupes homogènes, permettant de capturer des relations complexes dans les données de consommation.</li>
            <li><strong>Régression à Vecteurs de Support (SVR)</strong> : Un modèle de régression robuste qui transforme les données d'entrée dans un espace de plus grande dimension afin de trouver une meilleure approximation des données non linéaires.</li>
        </ul>

        <p>Les prévisions générées par ces modèles sont accompagnées d'intervalles de confiance à 95 %, offrant ainsi une évaluation de l'incertitude associée à chaque estimation. Ces informations sont cruciales pour une prise de décision éclairée, notamment en matière de planification des ressources et de stratégie de gestion de l'eau.</p>

        <p>Grâce à cette approche, les utilisateurs peuvent visualiser les tendances à court et moyen terme de consommation d'eau pour chaque opérateur, facilitant ainsi la gestion et l'optimisation des ressources.</p>
    """, unsafe_allow_html=True)

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

    st.markdown("""
    <hr>
    <h3>🔚 <strong>4. Conclusion</strong></h3>

    <p>Ce tableau de bord 💧 constitue un outil stratégique essentiel pour le suivi, l’analyse et l’anticipation de la consommation d’eau dans les ports marocains. En combinant des visualisations dynamiques avec des modèles de prévision performants, il permet non seulement d’observer les tendances passées, mais aussi d’appuyer les décisions futures en matière de gestion des ressources hydriques.</p>

    <p>Sa structure modulaire, sa capacité à comparer les opérateurs et à intégrer des scénarios prospectifs en font une solution complète, évolutive et adaptée aux besoins des gestionnaires publics, des opérateurs privés et des institutions de régulation. Il contribue ainsi à renforcer la transparence, l'efficacité opérationnelle et la planification durable dans le secteur de l'eau portuaire.</p>
    """, unsafe_allow_html=True)

    # ✅ Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: gray; font-style: italic; font-size: 14px;'>© Mai 2025 | Tableau de bord développé par M. Bougantouche & M. Bouceta</p>",
        unsafe_allow_html=True
    )
elif page == "Enquête Terrain":
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from io import StringIO
    from scipy.stats import pearsonr
    import plotly.figure_factory as ff

    # Données brutes définies globalement
    data = """Operateur,1. Votre entreprise dispose-t-elle de douches pour le personnel ?,2. Votre entreprise dispose-t-elle d’un restaurant pour le personnel?,3. Votre entreprise dispose-t-elle d’un jardin/système d’arrosage de plantes ?,4. Votre entreprise dispose-t-elle d’une ou plusieurs lave-vaisselle?,5. Votre entreprise dispose-t-elle d’un ou de plusieurs lave linge ?,6. De combien de toilettes disposez-vous dans votre bâtiments ?,7. Combien de personnes fréquentent approximativement vos locaux quotidiennement (personnel et visiteurs éventuels) ?
    ANP,oui,non,oui,0,0,15,200
    GLACIERES DU PORT,oui,non,non,0,0,2,10
    Marsa Maroc,oui,non,non,0,0,90,1800
    OCP,oui,non,non,0,0,64,550
    ONP,non,non,non,0,0,10,100
    somaport,oui,non,non,0,0,25,800
    Mana mesine,oui,non,non,0,0,2,10
    SOGAP,oui,non,non,0,0,2,30
    """

    # Titre principal
    st.markdown('<div class="main-title">📊 Analyse de la Consommation Domestique d\'Eau au Port de Casablanca : Résultats de l\'Enquête Terrain</div>', unsafe_allow_html=True)
    # Date et heure actuelles en français
    current_datetime = datetime.now().strftime("%d %B %Y %H:%M:%S")

    # Affichage dans Streamlit
    st.markdown(f"<p><strong>Dernière mise à jour :</strong> {current_datetime}</p>", unsafe_allow_html=True)

    # Introduction
    st.markdown("""
    <div class="intro-box">
        <h4>Introduction</h4>
        <p>La gestion durable des ressources en eau est un enjeu majeur dans les zones portuaires, où les activités économiques
et humaines exercent une pression croissante sur cette ressource essentielle. Le Port de Casablanca, en tant que hub
économique majeur du Maroc, concentre une diversité d’entreprises dont les activités influencent directement la consommation
d’eau domestique. Afin de mieux comprendre ces dynamiques, une enquête terrain a été menée pour collecter des données détaillées sur
les équipements liés à la consommation d’eau domesstique dans ces entreprises. 

Objectifs de l'étude :
1. Révéler des tendances statistiques : identifier les patterns de consommation domestique d’eau par types d’équipements et opérateurs.
2. Proposer des visualisations interactives : offrir des outils graphiques intuitifs permettant aux utilisateurs d’explorer
   les données de manière dynamique.
3. Identifier des corrélations : détecter des liens potentiels entre la fréquentation, les types d’équipements et les volumes
   d’eau consommés, afin de mieux orienter les stratégies de gestion de l’eau.</p>
    </div>
    """, unsafe_allow_html=True)

    # Vue d'Ensemble des Données
    st.markdown('<div class="section-header">Vue d\'Ensemble des Données</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Données Brutes de l\'Enquête</div>', unsafe_allow_html=True)
    st.dataframe(pd.read_csv(StringIO(data)), use_container_width=True)

    # CSS pour le style
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .section-header {
        font-size: 1.8em;
        color: #1565C0;
        border-bottom: 2px solid #BBDEFB;
        padding-bottom: 0.2em;
    }
    .subheader {
        font-size: 1.4em;
        color: #1976D2;
        margin-top: 0.8em;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1em;
        border-radius: 10px;
        margin-bottom: 1em;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .graph-comment {
        background-color: #F8FAFD;
        padding: 1em;
        margin: 0.8em 0;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
    }
    .graph-comment h4 {
        color: #1565C0;
        font-size: 1.2em;
        margin-bottom: 0.4em;
    }
    .graph-comment ul {
        list-style-type: none;
        padding-left: 0;
    }
    .graph-comment li {
        margin-bottom: 0.5em;
        padding-left: 1.2em;
        position: relative;
    }
    .graph-comment li::before {
        content: '➤';
        color: #1E88E5;
        position: absolute;
        left: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Chargement et Prétraitement des Données ---
    @st.cache_data
    def load_and_preprocess_data():
        df = pd.read_csv(StringIO(data))
        noms_colonnes = {
            '1. Votre entreprise dispose-t-elle de douches pour le personnel ?': 'Douches',
            '2. Votre entreprise dispose-t-elle d’un restaurant pour le personnel?': 'Restaurant',
            '3. Votre entreprise dispose-t-elle d’un jardin/système d’arrosage de plantes ?': 'Jardin_Arrosage',
            '4. Votre entreprise dispose-t-elle d’une ou plusieurs lave-vaisselle?': 'LaveVaisselle',
            '5. Votre entreprise dispose-t-elle d’un ou de plusieurs lave linge ?': 'LaveLinge',
            '6. De combien de toilettes disposez-vous dans votre bâtiments ?': 'NbToilettes',
            '7. Combien de personnes fréquentent approximativement vos locaux quotidiennement (personnel et visiteurs éventuels) ?': 'NbPersonnesQuotidien'
        }
        df.rename(columns=noms_colonnes, inplace=True)
        df.set_index('Operateur', inplace=True)

        colonnes_oui_non = ['Douches', 'Restaurant', 'Jardin_Arrosage']
        for col in colonnes_oui_non:
            df[col] = df[col].str.strip().str.lower().map({'oui': 1, 'non': 0}).fillna(0)
        df['LaveVaisselle'] = pd.to_numeric(df['LaveVaisselle'], errors='coerce').fillna(0).astype(int)
        df['LaveLinge'] = pd.to_numeric(df['LaveLinge'], errors='coerce').fillna(0).astype(int)
        return df

    df = load_and_preprocess_data()

    # --- Exploration Visuelle des Données ---
    st.markdown('<div class="section-header">Exploration Visuelle des Données</div>', unsafe_allow_html=True)

    # 1. Analyse des Équipements (Catégoriques)
    st.markdown('<div class="subheader">Analyse des Équipements (Catégoriques)</div>', unsafe_allow_html=True)
    show_cat_bar = st.checkbox("Diagramme en barre", value=True, key='cat_bar')
    show_cat_pie = st.checkbox("Diagramme en cercle", value=True, key='cat_pie')

    # Analyse des équipements catégoriques
    variables_categorielles = ['Douches', 'Restaurant', 'Jardin_Arrosage', 'LaveVaisselle', 'LaveLinge']
    constantes = [col for col in variables_categorielles if df[col].nunique() == 1]

    # Équipements non présents
    if constantes and show_cat_pie:
        st.markdown('<div class="subheader">Équipements Non Présents</div>', unsafe_allow_html=True)
        for var in constantes:
            fig_donut = go.Figure(data=[
                go.Pie(labels=['Non (0)'], values=[len(df)], hole=0.5, textinfo='percent+label', marker=dict(colors=['#FF9999']))
            ])
            fig_donut.update_layout(title=f'Répartition de {var}', height=300, annotations=[dict(text='100%', x=0.5, y=0.5, font_size=16, showarrow=False)])
            st.plotly_chart(fig_donut, use_container_width=True)
            st.markdown(f"""
            <div class="graph-comment">
                <h4>Répartition de {var}</h4>
                <ul>
                    <li>100% (8/8 entreprises) sans {var.lower()}.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Distribution des équipements actifs
    st.markdown('<div class="subheader">Distribution des Équipements Actifs</div>', unsafe_allow_html=True)
    cat_summary = pd.DataFrame({
        'Équipement': variables_categorielles,
        'Pourcentage Oui (%)': [round(df[col].mean() * 100, 1) for col in variables_categorielles],
        'Nombre Oui': [df[col].sum() for col in variables_categorielles]
    })
    st.table(cat_summary)
    st.markdown("""
    <div class="graph-comment">
        <h4>Synthèse Statistique</h4>
        <ul>
            <li>Douches : 75% (6/8 entreprises), équipement dominant.</li>
            <li>Jardin/Arrosage : 12.5% (1/8, ANP), usage marginal.</li>
            <li>Autres équipements : absents, non prioritaires.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    for var in ['Douches', 'Jardin_Arrosage']:
        value_counts_df = df[var].value_counts().reset_index()
        value_counts_df.columns = ['Valeur', 'Nombre']
        oui_count = value_counts_df[value_counts_df['Valeur'] == 1]['Nombre'].iloc[0] if 1 in value_counts_df['Valeur'].values else 0
        col1, col2 = st.columns(2)
        if show_cat_bar:
            with col1:
                fig_bar = px.bar(value_counts_df, x='Valeur', y='Nombre', title=f'Distribution de {var} (Barres)', color='Valeur', color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_bar.update_layout(xaxis_title='Valeur (0 = Non, 1 = Oui)', yaxis_title='Nombre', showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown(f"""
                <div class="graph-comment">
                    <h4>Distribution de {var} (Barres)</h4>
                    <ul>
                        <li>{oui_count}/8 ({round(oui_count/8*100, 1)}%) ont {var.lower()}.</li>
                        <li>{'Priorité sanitaire.' if var == 'Douches' else 'Spécifique à ANP.'}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        if show_cat_pie:
            with col2:
                fig_pie = px.pie(value_counts_df, values='Nombre', names='Valeur', title=f'Répartition de {var} (Cercle)', color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown(f"""
                <div class="graph-comment">
                    <h4>Répartition de {var} (Cercle)</h4>
                    <ul>
                        <li>{round(oui_count/8*100, 1)}% avec, {100-round(oui_count/8*100, 1)}% sans.</li>
                        <li>{'Forte adoption.' if var == 'Douches' else 'Usage minoritaire.'}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # 2. Statistiques des Variables Numériques
    st.markdown('<div class="subheader">Statistiques des Variables Numériques</div>', unsafe_allow_html=True)
    show_quant_hist = st.checkbox("Diagramme en barre (Histogramme)", value=True, key='quant_hist')
    show_quant_violin = st.checkbox("Diagramme en tuyaux", value=True, key='quant_violin')
    show_quant_box = st.checkbox("Boxplot", value=True, key='quant_box')

    # Analyse des variables quantitatives
    variables_quantitatives = ['NbToilettes', 'NbPersonnesQuotidien']
    stats_summary = pd.DataFrame({
        'Statistique': ['Moyenne', 'Écart-type', 'Médiane', 'Min', 'Max', 'CV (%)'],
        'NbToilettes': [round(df['NbToilettes'].mean(), 2), round(df['NbToilettes'].std(), 2), df['NbToilettes'].median(), df['NbToilettes'].min(), df['NbToilettes'].max(), round(df['NbToilettes'].std() / df['NbToilettes'].mean() * 100, 1)],
        'NbPersonnesQuotidien': [round(df['NbPersonnesQuotidien'].mean(), 2), round(df['NbPersonnesQuotidien'].std(), 2), df['NbPersonnesQuotidien'].median(), df['NbPersonnesQuotidien'].min(), df['NbPersonnesQuotidien'].max(), round(df['NbPersonnesQuotidien'].std() / df['NbPersonnesQuotidien'].mean() * 100, 1)]
    })
    st.table(stats_summary)
    st.markdown("""
    <div class="graph-comment">
        <h4>Synthèse Statistique</h4>
        <ul>
            <li>NbToilettes : Moyenne 26.25, médiane 12.5, CV 125.8%. Distribution asymétrique (2 à 90).</li>
            <li>NbPersonnesQuotidien : Moyenne 437.5, médiane 150, CV 144.6%. Valeur extrême à 1800.</li>
            <li>Forte variabilité, influencée par Marsa Maroc.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    for var in variables_quantitatives:
        with st.expander(f"📈 Analyse de {var}", expanded=True):
            stats = df[var].describe()
            cv = df[var].std() / df[var].mean() * 100
            stats_df = pd.DataFrame({
                'Statistique': ['Moyenne', 'Écart-type', 'Médiane', 'Min', 'Max', 'CV (%)'],
                'Valeur': [round(stats['mean'], 2), round(stats['std'], 2), stats['50%'], stats['min'], stats['max'], round(cv, 2)]
            })
            st.markdown("**Statistiques Descriptives**")
            st.table(stats_df)
            st.markdown(f"""
            <div class="graph-comment">
                <h4>Résumé de {var}</h4>
                <ul>
                    <li>Moyenne ({round(stats['mean'], 2)}) > médiane ({stats['50%']}), distribution asymétrique.</li>
                    <li>CV {round(cv, 2)}% : forte dispersion.</li>
                    <li>Étendue : {int(stats['min'])} à {int(stats['max'])}.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            if show_quant_hist:
                fig_hist = px.histogram(df, x=var, nbins=10, title=f'Distribution de {var} (Histogramme)', color_discrete_sequence=['#1E88E5'])
                fig_hist.update_layout(xaxis_title=var, yaxis_title='Fréquence', showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown(f"""
                <div class="graph-comment">
                    <h4>Distribution de {var} (Histogramme)</h4>
                    <ul>
                        <li>Concentration autour de {stats['50%']}, queue vers {stats['max']}.</li>
                        <li>{'Marsa Maroc (90) extrême.' if var == 'NbToilettes' else 'Marsa Maroc (1800) extrême.'}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            if show_quant_violin:
                fig_violin = go.Figure(data=go.Violin(y=df[var], box_visible=True, meanline_visible=True, fillcolor='#BBDEFB', points='all'))
                fig_violin.update_layout(title=f'Distribution de {var}', yaxis_title=var, showlegend=False)
                st.plotly_chart(fig_violin, use_container_width=True)
                st.markdown(f"""
                <div class="graph-comment">
                    <h4>Distribution de {var} (Violon)</h4>
                    <ul>
                        <li>Médiane {stats['50%']}, forte densité dans l'interquartile.</li>
                        <li>Valeur extrême à {stats['max']} (Marsa Maroc).</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            if show_quant_box:
                fig_box = go.Figure(data=go.Box(y=df[var], boxpoints='all', jitter=0.3, pointpos=-1.8, fillcolor='#BBDEFB'))
                fig_box.update_layout(title=f'Distribution de {var} (Boxplot)', yaxis_title=var, showlegend=False)
                fig_box.update_traces(marker=dict(color='#1E88E5'))
                st.plotly_chart(fig_box, use_container_width=True)
                st.markdown(f"""
                <div class="graph-comment">
                    <h4>Distribution de {var} (Boxplot)</h4>
                    <ul>
                        <li>Médiane {stats['50%']}, interquartile [{stats['25%']}, {stats['75%']}].</li>
                        <li>Valeur extrême à {stats['max']} (Marsa Maroc).</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            Q1, Q3 = stats['25%'], stats['75%']
            IQR = Q3 - Q1
            outliers = df[(df[var] < Q1 - 1.5 * IQR) | (df[var] > Q3 + 1.5 * IQR)][var]
            st.markdown("**Valeurs Aberrantes**")
            st.write(f"Entreprises : {', '.join(outliers.index.tolist()) if not outliers.empty else 'Aucune'}")
            if not outliers.empty:
                st.write(f"Valeurs : {outliers.values.tolist()}")

    # 3. Analyse de Corrélation Bivariée
    st.markdown('<div class="subheader">Analyse de Corrélation Bivariée</div>', unsafe_allow_html=True)
    show_corr_scatter = st.checkbox("Nuage de Points", value=True, key='corr_scatter')
    show_corr_heatmap = st.checkbox("Matrice de Corrélation", value=True, key='corr_heatmap')

    # Analyse de corrélation
    st.markdown('<div class="subheader">Relation entre Toilettes et Fréquentation</div>', unsafe_allow_html=True)
    corr, p_value = pearsonr(df['NbToilettes'], df['NbPersonnesQuotidien'])
    st.markdown(f"""
    <div class="graph-comment">
        <h4>Synthèse Statistique</h4>
        <ul>
            <li>Corrélation de Pearson : {round(corr, 3)}, liaison forte et positive.</li>
            <li>P-valeur : {round(p_value, 3)}, {'significative' if p_value < 0.05 else 'non significative'}.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if show_corr_scatter:
        fig_scatter = px.scatter(df, x='NbToilettes', y='NbPersonnesQuotidien', text=df.index, title='Relation entre Toilettes et Fréquentation (Nuage)', color_discrete_sequence=['#1E88E5'])
        fig_scatter.update_traces(marker=dict(size=12))
        fig_scatter.update_layout(showlegend=False)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown(f"""
        <div class="graph-comment">
            <h4>Nuage de Points</h4>
            <ul>
                <li>Forte liaison : plus de fréquentation, plus de toilettes.</li>
                <li>Valeur extrême : Marsa Maroc (90 toilettes, 1800 personnes).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if show_corr_heatmap:
        corr_matrix = df[variables_quantitatives].corr()
        fig_heatmap = ff.create_annotated_heatmap(z=corr_matrix.values, x=variables_quantitatives, y=variables_quantitatives, colorscale='Blues', annotation_text=corr_matrix.round(2).values)
        fig_heatmap.update_layout(title='Matrice de Corrélation', width=500, height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.markdown(f"""
        <div class="graph-comment">
            <h4>Matrice de Corrélation</h4>
            <ul>
                <li>Corrélation {round(corr, 2)} : forte liaison entre fréquentation et nombre de toilettes.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.header("Principaux enseignements")
    st.write("""
    L’enquête menée auprès de huit opérateurs du port de Casablanca vise à décrypter les usages domestiques de l’eau, 
    en examinant les équipements liés à la consommation (sanitaires, restauration, arrosage, entretien) et la fréquentation 
    quotidienne.
    """)

    # Distribution des Équipements
    st.write("""
    Concernant les variables catégoriques, l’analyse des équipements révèle une adoption inégale des installations consommatrices d’eau :
    - **Douches** : Présentes chez 75% des opérateurs (6/8, dont ANP, Marsa Maroc, OCP), elles constituent l’équipement 
    sanitaire le plus courant, reflétant des besoins d’hygiène pour le personnel portuaire.
    - **Jardin/Arrosage** : Uniquement chez ANP (12.5%, 1/8), cet usage est marginal, suggérant une faible priorité.
    - **Restaurants, Lave-vaisselle, Lave-linge** : Absents chez tous les opérateurs (100%, 8/8).


    """)

    # Statistiques des Variables Numériques
    st.write("""
    Les variables quantitatives, nombre de toilettes (NbToilettes) et fréquentation quotidienne (NbPersonnesQuotidien), 
    présentent une forte variabilité, influencée par la taille des opérateurs :

    **Nombre de Toilettes** :
    - Moyenne : 26,25 toilettes, médiane : 12,5, coefficient de variation (CV) : 125,8%.
    - Étendue : 2 (petits opérateurs comme Mana Mesine) à 90 (Marsa Maroc).
    - Distribution asymétrique, avec une queue vers les grandes valeurs. Marsa Maroc (90 toilettes) est une valeur extrême, 
    reflétant son envergure.

    **Fréquentation Quotidienne** :
    - Moyenne : 437,5 personnes, médiane : 150, CV : 144,6%.
    - Étendue : 10 (Mana Mesine, Glacières du Port) à 1800 (Marsa Maroc).
    - Distribution fortement asymétrique, dominée par Marsa Maroc, qui concentre une fréquentation exceptionnelle.

    La forte dispersion (CV élevé) et les valeurs extrêmes soulignent l’hétérogénéité des opérateurs portuaires, avec des 
    implications directes sur la consommation d’eau liée aux sanitaires.
    """)


    st.write("""
    L’étude de la relation entre le nombre de toilettes et la fréquentation quotidienne révèle une liaison significative :
    - **Corrélation de Pearson** : 0,944, indiquant une relation linéaire forte et positive.
    - **P-valeur** : 0,0002, confirmant la significativité statistique (p < 0,05).

    Cette corrélation suggère que la fréquentation est un déterminant majeur de la demande en installations sanitaires, 
    et donc de la consommation domestique d’eau.
    """)
    # ✅ Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
    "<p style='text-align: center; color: gray; font-style: italic; font-size: 14px;'>© Mai 2025 | Dashboard développé par M. Bougantouche & M. Bouceta</p>", unsafe_allow_html=True)