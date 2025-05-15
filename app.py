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
st.set_page_config(layout="wide", page_title= "Global Water Consumption Dashboard")

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
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from scipy.stats import pearsonr
    import plotly.figure_factory as ff

    # Configuration de la page pour un design amélioré
        # Introduction
    st.markdown("""
    <div class="intro-box">
        <h4>Introduction</h4>
        <p>Cette interface présente une analyse approfondie des résultats d'une enquête menée auprès des entreprises enquêtées, visant à évaluer leurs infrastructures et équipements. L'étude se concentre sur les installations sanitaires (<strong>toilettes</strong>, <strong>douches</strong>), les commodités de restauration (<strong>restaurants</strong>), les aménagements extérieurs (<strong>jardins</strong> ou <strong>systèmes d'arrosage</strong>), ainsi que les équipements d'entretien (<strong>lave-vaisselle</strong>, <strong>lave-linge</strong>). Elle examine également la fréquentation quotidienne des locaux par le personnel et les visiteurs.</p>
        <p>L'objectif principal est de fournir une compréhension claire des tendances et des relations entre ces variables à travers :</p>
        <ul>
            <li>Des <strong>statistiques descriptives</strong> détaillées pour chaque équipement et variable quantitative.</li>
            <li>Des <strong>visualisations interactives</strong>, incluant histogrammes, diagrammes en violon et nuages de points, pour une exploration visuelle des données.</li>
            <li>Des <strong>analyses univariées et bivariées</strong>, mettant en lumière les distributions, les variabilités et les corrélations potentielles.</li>
        </ul>
        <p>Les sections suivantes offrent une exploration structurée des données, avec des interprétations rigoureuses pour soutenir la prise de décision.</p>
    </div>
    """, unsafe_allow_html=True)
    # CSS personnalisé pour améliorer le style des commentaires
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.5em;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 0.5em;
            font-weight: bold;
        }
        .section-header {
            font-size: 1.8em;
            color: #1565C0;
            border-bottom: 2px solid #BBDEFB;
            padding-bottom: 0.2em;
            margin-top: 1em;
        }
        .subheader {
            font-size: 1.4em;
            color: #1976D2;
            margin-top: 0.8em;
            font-weight: 600;
        }
        .info-box {
            background-color: #E3F2FD;
            padding: 1em;
            border-radius: 10px;
            margin-bottom: 1em;
            border: 1px solid #BBDEFB;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border-radius: 5px;
            padding: 0.5em 1em;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #1565C0;
            color: #E3F2FD;
        }
        .graph-comment, .table-comment {
            background-color: #F8FAFD;
            padding: 1.2em;
            margin: 0.8em 0;
            border-radius: 8px;
            border-left: 6px solid #1E88E5;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            font-size: 1.1em;
            line-height: 1.5;
            color: #333;
        }
        .graph-comment h4, .table-comment h4 {
            color: #1565C0;
            font-size: 1.3em;
            margin-bottom: 0.5em;
            display: flex;
            align-items: center;
        }
        .graph-comment h4::before, .table-comment h4::before {
            content: '📊 ';
            margin-right: 0.3em;
        }
        .graph-comment ul, .table-comment ul {
            list-style-type: none;
            padding-left: 0;
        }
        .graph-comment li, .table-comment li {
            margin-bottom: 0.6em;
            position: relative;
            padding-left: 1.5em;
        }
        .graph-comment li::before, .table-comment li::before {
            content: '➤';
            color: #1E88E5;
            position: absolute;
            left: 0;
        }
        .graph-comment strong, .table-comment strong {
            color: #D81B60;
            font-weight: 600;
        }
        .expander-content {
            background-color: #FFFFFF;
            padding: 1em;
            border-radius: 5px;
            border: 1px solid #E0E0E0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Titre principal
    st.markdown('<div class="main-title">📊 Analyse d\'Enquête sur les Équipements d\'Entreprise</div>', unsafe_allow_html=True)
    st.markdown("Une application interactive pour explorer les données d'enquête, visualiser les tendances et effectuer des analyses statistiques avancées.")

    # --- 1. Chargement et Prétraitement des Données ---
    @st.cache_data
    def load_and_preprocess_data():
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

        # Prétraitement
        colonnes_oui_non = ['Douches', 'Restaurant', 'Jardin_Arrosage']
        for col in colonnes_oui_non:
            df[col] = df[col].str.strip().str.lower().map({'oui': 1, 'non': 0})
            if df[col].isna().any():
                df[col].fillna(0, inplace=True)

        df['LaveVaisselle'] = pd.to_numeric(df['LaveVaisselle'], errors='coerce').fillna(0).astype(int)
        df['LaveLinge'] = pd.to_numeric(df['LaveLinge'], errors='coerce').fillna(0).astype(int)
        return df

    # Charger les données
    df = load_and_preprocess_data()

    # --- 2. Aperçu des Données ---
    st.markdown('<div class="section-header">1. Aperçu des Données</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="subheader">Données Initiales (Avant Renommage)</div>', unsafe_allow_html=True)
        data_orig_str = """Operateur,1. Votre entreprise dispose-t-elle de douches pour le personnel ?,2. Votre entreprise dispose-t-elle d’un restaurant pour le personnel?,3. Votre entreprise dispose-t-elle d’un jardin/système d’arrosage de plantes ?,4. Votre entreprise dispose-t-elle d’une ou plusieurs lave-vaisselle?,5. Votre entreprise dispose-t-elle d’un ou de plusieurs lave linge ?,6. De combien de toilettes disposez-vous dans votre bâtiments ?,7. Combien de personnes fréquentent approximativement vos locaux quotidiennement (personnel et visiteurs éventuels) ?
    ANP,oui,non,oui,0,0,15,200
    GLACIERES DU PORT,oui,non,non,0,0,2,10
    Marsa Maroc,oui,non,non,0,0,90,1800
    OCP,oui,non,non,0,0,64,550
    ONP,non,non,non,0,0,10,100
    somaport,oui,non,non,0,0,25,800
    Mana mesine,oui,non,non,0,0,2,10
    SOGAP,oui,non,non,0,0,2,30
    """
        df_orig_display = pd.read_csv(StringIO(data_orig_str))
        st.dataframe(df_orig_display, use_container_width=True)

        with st.expander("🔍 Informations sur les Types de Données"):
            buffer = StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

    # --- 3. Analyse Univariée ---
    st.markdown('<div class="section-header">2. Analyse Univariée</div>', unsafe_allow_html=True)

    # Variables catégoriques
    with st.container():
        st.markdown('<div class="subheader">Distribution des Équipements (Variables Catégorielles)</div>', unsafe_allow_html=True)
        variables_categorielles_encodees = ['Douches', 'Restaurant', 'Jardin_Arrosage', 'LaveVaisselle', 'LaveLinge']
        colonnes_a_exclure_plot = [col for col in variables_categorielles_encodees if df[col].nunique() == 1]

        # Analyse des variables constantes
        if colonnes_a_exclure_plot:
            st.markdown('<div class="subheader">Analyse des Variables Constantes</div>', unsafe_allow_html=True)
            for var in colonnes_a_exclure_plot:
                # Créer un DataFrame pour le demi-donut
                value_counts_df = pd.DataFrame({'Valeur': [0], 'Nombre': [len(df)]})
                # Demi-donut chart
                fig_donut = go.Figure(data=[
                    go.Pie(
                        labels=['Non (0)'],
                        values=[len(df)],
                        hole=0.5,
                        pull=[0.1],
                        direction='clockwise',
                        rotation=90,
                        sort=False,
                        textinfo='percent+label',
                        textposition='inside',
                        marker=dict(colors=['#FF9999'])
                    )
                ])
                fig_donut.update_layout(
                    title=f'Répartition de {var} (Demi-Donut)',
                    showlegend=True,
                    height=400,
                    annotations=[dict(text='100%', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig_donut, use_container_width=True)
                # Commentaire détaillé
                st.markdown(f"""
                <div class="graph-comment">
                    <h4>Interprétation de {var}</h4>
                    <ul>
                        <li><strong>Constante à 0</strong>: Toutes les 8 entreprises ont une valeur de 0 pour {var}, indiquant une absence totale de {'lave-vaisselle' if var == 'LaveVaisselle' else 'lave-linge'} dans leurs locaux.</li>
                        <li><strong>Graphique</strong>: Le demi-donut montre que 100% des entreprises (8 sur 8) n'ont pas cet équipement, visualisé par une seule section rouge.</li>
                        <li><strong>Contexte</strong>: Cette absence peut être liée au type d'entreprises interrogées (ex. industrielles ou portuaires), où les équipements de lavage ne sont pas nécessaires ou sont externalisés.</li>
                        <li><strong>Implications</strong>:
                            <ul>
                                <li><strong>Infrastructure</strong>: Les entreprises privilégient d'autres équipements, comme les douches (75% de présence), probablement plus pertinentes pour le bien-être du personnel.</li>
                                <li><strong>Coût et pertinence</strong>: L'installation et l'entretien de {'lave-vaisselle' if var == 'LaveVaisselle' else 'lave-linge'} pourraient être jugés non prioritaires ou trop coûteux.</li>
                                <li><strong>Homogénéité</strong>: La constance de la valeur 0 suggère un consensus parmi les entreprises, peut-être dû à des normes sectorielles ou à des contraintes logistiques.</li>
                            </ul>
                        </li>
                        <li><strong>Comparaison</strong>:
                            <ul>
                                <li>Par rapport à <code>Douches</code> (6 entreprises sur 8, 75%) ou <code>Jardin_Arrosage</code> (1 entreprise, 12.5%), {var} est complètement absent, soulignant une différence marquée dans les priorités d'équipement.</li>
                                <li>Similaire à <code>Restaurant</code> (0% de présence), mais contrairement à <code>Restaurant</code>, {var} pourrait être moins attendu dans un contexte industriel.</li>
                            </ul>
                        </li>
                        <li><strong>Conclusion</strong>: L'absence de {var} reflète probablement une inadéquation avec les besoins opérationnels ou les budgets des entreprises interrogées, contrairement aux équipements sanitaires plus répandus.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        # Variables non constantes
        st.markdown('<div class="subheader">Distribution des Variables Non Constantes</div>', unsafe_allow_html=True)
        # Comparaison des variables catégoriques
        st.markdown("**Comparaison des Équipements**")
        cat_summary = pd.DataFrame({
            'Équipement': variables_categorielles_encodees,
            'Pourcentage Oui (%)': [round(df[col].mean() * 100, 1) if col not in colonnes_a_exclure_plot else 0 for col in variables_categorielles_encodees],
            'Nombre Oui': [df[col].sum() if col not in colonnes_a_exclure_plot else 0 for col in variables_categorielles_encodees]
        })
        st.table(cat_summary)
        st.markdown(f"""
        <div class="table-comment">
            <h4>Interprétation du Tableau</h4>
            <ul>
                <li><strong>Douches</strong>: {cat_summary[cat_summary['Équipement'] == 'Douches']['Pourcentage Oui (%)'].iloc[0]}% des entreprises (6 sur 8) disposent de douches, ce qui en fait l'équipement le plus courant.</li>
                <li><strong>Restaurant</strong>: Aucun restaurant n'est présent (0%), indiquant une absence totale de cet équipement.</li>
                <li><strong>Jardin_Arrosage</strong>: Seulement {cat_summary[cat_summary['Équipement'] == 'Jardin_Arrosage']['Pourcentage Oui (%)'].iloc[0]}% (1 entreprise, ANP) disposent d'un jardin ou système d'arrosage, ce qui est rare.</li>
                <li><strong>LaveVaisselle et LaveLinge</strong>: Aucun (0%), montrant une absence complète de ces équipements.</li>
                <li><strong>Comparaison</strong>: Les douches sont nettement plus répandues que les autres équipements, tandis que les restaurants et les systèmes de lavage sont inexistants, suggérant des priorités différentes dans les infrastructures des entreprises.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        for var in [col for col in variables_categorielles_encodees if col not in colonnes_a_exclure_plot]:
            # Compute value counts and reset index
            value_counts_df = df[var].value_counts().reset_index()
            value_counts_df.columns = ['Valeur', 'Nombre']
            # Bar plot
            col1, col2 = st.columns(2)
            with col1:
                fig_bar = px.bar(value_counts_df, x='Valeur', y='Nombre',
                                labels={'Valeur': 'Valeur (0 = Non, 1 = Oui)', 'Nombre': 'Nombre'},
                                title=f'Distribution de {var} (Barres)',
                                color='Valeur', color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_bar.update_layout(xaxis_title="Valeur (0 = Non, 1 = Oui)", yaxis_title="Nombre d'entreprises", showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
                # Commentaire pour le graphique en barres
                oui_count = value_counts_df[value_counts_df['Valeur'] == 1]['Nombre'].iloc[0] if 1 in value_counts_df['Valeur'].values else 0
                non_count = value_counts_df[value_counts_df['Valeur'] == 0]['Nombre'].iloc[0] if 0 in value_counts_df['Valeur'].values else 0
                oui_pct = round(oui_count / (oui_count + non_count) * 100, 1)
                non_pct = round(non_count / (oui_count + non_count) * 100, 1)
                st.markdown(f"""
                <div class="graph-comment">
                    <h4>Interprétation du Graphique en Barres ({var})</h4>
                    <ul>
                        <li><strong>Répartition</strong>: Le diagramme montre la répartition de l'équipement "{var}" parmi les 8 entreprises.</li>
                        <li><strong>Oui (1)</strong>: {oui_count} entreprises ({oui_pct}%) disposent de {var.lower()}.</li>
                        <li><strong>Non (0)</strong>: {non_count} entreprises ({non_pct}%) n'en disposent pas.</li>
                        <li><strong>Analyse</strong>: {'Les douches sont majoritaires, reflétant une priorité dans les infrastructures.' if var == 'Douches' else 'Les jardins sont très rares, ANP étant une exception.' if var == 'Jardin_Arrosage' else 'Aucun restaurant, ce qui est cohérent avec les infrastructures industrielles.'}</li>
                        <li><strong>Comparaison</strong>: Par rapport aux autres équipements, {var} est {'le plus courant' if var == 'Douches' else 'beaucoup moins répandu' if var == 'Jardin_Arrosage' else 'totalement absent'}.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Pie chart
            with col2:
                fig_pie = px.pie(value_counts_df, values='Nombre', names='Valeur',
                                title=f'Distribution de {var} (Secteurs)',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_traces(textinfo='percent+label', textposition='inside')
                fig_pie.update_layout(showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True)
                # Commentaire pour le graphique en secteurs
                st.markdown(f"""
                <div class="graph-comment">
                    <h4>Interprétation du Graphique en Secteurs ({var})</h4>
                    <ul>
                        <li><strong>Proportion</strong>: Ce diagramme illustre la proportion des entreprises avec ou sans {var.lower()}.</li>
                        <li><strong>Oui (1)</strong>: {oui_pct}% des entreprises, soit {oui_count} sur 8.</li>
                        <li><strong>Non (0)</strong>: {non_pct}% des entreprises, soit {non_count} sur 8.</li>
                        <li><strong>Analyse</strong>: La visualisation met en évidence {'une forte adoption des douches' if var == 'Douches' else 'la rareté des jardins' if var == 'Jardin_Arrosage' else 'l\'absence totale de restaurants'}.</li>
                        <li><strong>Comparaison</strong>: Par rapport à {'Douches (75% Oui)' if var != 'Douches' else 'Jardin_Arrosage (12.5% Oui)'}, {var} montre {'une adoption bien moindre' if var != 'Douches' else 'la plus forte adoption'}.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # Variables quantitatives
    with st.container():
        st.markdown('<div class="subheader">Distribution des Variables Quantitatives</div>', unsafe_allow_html=True)
        variables_quantitatives = ['NbToilettes', 'NbPersonnesQuotidien']

        # Comparaison des variables quantitatives
        st.markdown("**Comparaison des Statistiques Descriptives**")
        stats_summary = pd.DataFrame({
            'Statistique': ['Moyenne', 'Écart-type', 'Médiane', 'Minimum', 'Maximum', 'Coefficient de variation (%)'],
            'NbToilettes': [
                round(df['NbToilettes'].mean(), 2),
                round(df['NbToilettes'].std(), 2),
                round(df['NbToilettes'].median(), 2),
                int(df['NbToilettes'].min()),
                int(df['NbToilettes'].max()),
                round(df['NbToilettes'].std() / df['NbToilettes'].mean() * 100, 1)
            ],
            'NbPersonnesQuotidien': [
                round(df['NbPersonnesQuotidien'].mean(), 2),
                round(df['NbPersonnesQuotidien'].std(), 2),
                round(df['NbPersonnesQuotidien'].median(), 2),
                int(df['NbPersonnesQuotidien'].min()),
                int(df['NbPersonnesQuotidien'].max()),
                round(df['NbPersonnesQuotidien'].std() / df['NbPersonnesQuotidien'].mean() * 100, 1)
            ]
        })
        st.table(stats_summary)
        st.markdown(f"""
        <div class="table-comment">
            <h4>Interprétation du Tableau</h4>
            <ul>
                <li><strong>NbToilettes</strong>:
                    <ul>
                        <li><strong>Moyenne</strong>: 26.25 toilettes, mais médiane à 12.5, indiquant une distribution fortement asymétrique à droite.</li>
                        <li><strong>Écart-type</strong>: 33.03, CV : 125.8% → Très forte variabilité, due à des valeurs extrêmes comme Marsa Maroc (90 toilettes).</li>
                        <li><strong>Étendue</strong>: 2 à 90 toilettes, montrant une grande disparité entre petites et grandes entreprises.</li>
                    </ul>
                </li>
                <li><strong>NbPersonnesQuotidien</strong>:
                    <ul>
                        <li><strong>Moyenne</strong>: 437.5 personnes, médiane : 150 → Asymétrie encore plus prononcée.</li>
                        <li><strong>Écart-type</strong>: 632.73, CV : 144.6% → Variabilité plus élevée que pour NbToilettes, avec Marsa Maroc (1800 personnes) comme valeur extrême.</li>
                        <li><strong>Étendue</strong>: 10 à 1800 personnes, reflétant des différences importantes dans la taille des entreprises.</li>
                    </ul>
                </li>
                <li><strong>Comparaison</strong>:
                    <ul>
                        <li>NbPersonnesQuotidien montre une variabilité relative plus élevée (CV 144.6% vs 125.8%), due à une plus grande amplitude (1790 vs 88).</li>
                        <li>Les deux variables sont asymétriques à droite, mais NbPersonnesQuotidien a une queue plus longue (max 1800 vs 90).</li>
                        <li>Les entreprises comme Marsa Maroc dominent les deux variables, suggérant une corrélation potentielle.</li>
                    </ul>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        for var in variables_quantitatives:
            with st.expander(f"📈 Analyse de {var}", expanded=True):
                # Statistiques descriptives
                stats = df[var].describe()
                cv = df[var].std() / df[var].mean() * 100
                stats_df = pd.DataFrame({
                    'Statistique': ['Nombre', 'Moyenne', 'Écart-type', 'Médiane', 'Minimum', 'Q1 (25%)', 'Q3 (75%)', 'Maximum', 'Coefficient de variation'],
                    'Valeur': [
                        int(stats['count']),
                        round(stats['mean'], 2),
                        round(stats['std'], 2),
                        round(stats['50%'], 2),
                        int(stats['min']),
                        round(stats['25%'], 2),
                        round(stats['75%'], 2),
                        int(stats['max']),
                        f"{round(cv, 2)}%"
                    ]
                })
                st.markdown("**Statistiques Descriptives**")
                st.table(stats_df)
                # Commentaire pour le tableau
                st.markdown(f"""
                <div class="table-comment">
                    <h4>Interprétation du Tableau ({var})</h4>
                    <ul>
                        <li><strong>Moyenne vs Médiane</strong>: La moyenne ({round(stats['mean'], 2)}) est {'bien supérieure' if stats['mean'] > stats['50%'] * 1.5 else 'supérieure'} à la médiane ({round(stats['50%'], 2)}), indiquant une distribution asymétrique à droite.</li>
                        <li><strong>Dispersion</strong>: L'écart-type ({round(stats['std'], 2)}) et le CV ({round(cv, 2)}%) montrent une {'forte' if cv > 50 else 'modérée' if cv > 20 else 'faible'} variabilité.</li>
                        <li><strong>Étendue</strong>: De {int(stats['min'])} à {int(stats['max'])}, soit une différence de {int(stats['max'] - stats['min'])} {'toilettes' if var == 'NbToilettes' else 'personnes'}.</li>
                        <li><strong>Quartiles</strong>: 50% des entreprises ont entre {round(stats['25%'], 2)} et {round(stats['75%'], 2)} {'toilettes' if var == 'NbToilettes' else 'personnes'}, montrant {'une concentration autour de petites valeurs' if stats['25%'] < stats['mean'] else 'une répartition plus équilibrée'}.</li>
                        <li><strong>Comparaison</strong>: Par rapport à {'NbPersonnesQuotidien' if var == 'NbToilettes' else 'NbToilettes'}, {var} a {'une variabilité moindre' if var == 'NbToilettes' else 'une variabilité plus élevée'} (CV {round(cv, 2)}% vs {round(df['NbPersonnesQuotidien'].std() / df['NbPersonnesQuotidien'].mean() * 100, 2) if var == 'NbToilettes' else round(df['NbToilettes'].std() / df['NbToilettes'].mean() * 100, 2)}%).</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                # Visualisations
                col1, col2 = st.columns(2)
                with col1:
                    fig_hist = px.histogram(df, x=var, nbins=10, title=f'Distribution de {var}',
                                            color_discrete_sequence=['#1E88E5'])
                    fig_hist.update_layout(xaxis_title=var, yaxis_title='Fréquence', showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    # Commentaire pour l'histogramme
                    max_freq = df[var].value_counts(bins=10).max()
                    st.markdown(f"""
                    <div class="graph-comment">
                        <h4>Interprétation de l'Histogramme ({var})</h4>
                        <ul>
                            <li><strong>Répartition</strong>: Cet histogramme montre la répartition de "{var}" parmi les 8 entreprises.</li>
                            <li><strong>Concentration</strong>: La majorité des entreprises ont des valeurs autour de {round(stats['50%'], 2)} {'toilettes' if var == 'NbToilettes' else 'personnes'}, avec une fréquence maximale de {max_freq} entreprises par intervalle.</li>
                            <li><strong>Asymétrie</strong>: La distribution est {'fortement' if stats['mean'] > stats['50%'] * 1.5 else ''} à droite, avec une queue vers {int(stats['max'])} (ex. Marsa Maroc : {90 if var == 'NbToilettes' else 1800}).</li>
                            <li><strong>Variabilité</strong>: La large étendue ({int(stats['max'] - stats['min'])}) reflète {'une forte hétérogénéité' if cv > 50 else 'une hétérogénéité modérée'}.</li>
                            <li><strong>Comparaison</strong>: Par rapport à {'NbPersonnesQuotidien' if var == 'NbToilettes' else 'NbToilettes'}, {var} montre {'une queue moins longue' if var == 'NbToilettes' else 'une queue plus longue'} (max {int(stats['max'])} vs {1800 if var == 'NbToilettes' else 90}).</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    fig_violin = go.Figure(data=go.Violin(y=df[var], box_visible=True, line_color='#1565C0',
                                                        meanline_visible=True, fillcolor='#BBDEFB', opacity=0.6,
                                                        points='all', pointpos=0, jitter=0.05))
                    fig_violin.update_layout(title=f'Diagramme en Violon de {var}', yaxis_title=var, showlegend=False)
                    st.plotly_chart(fig_violin, use_container_width=True)
                    # Commentaire pour le violon
                    st.markdown(f"""
                    <div class="graph-comment">
                        <h4>Interprétation du Diagramme en Violon ({var})</h4>
                        <ul>
                            <li><strong>Densité</strong>: Ce diagramme montre la densité et la répartition de "{var}".</li>
                            <li><strong>Concentration</strong>: La largeur maximale autour de {round(stats['50%'], 2)} indique une forte concentration des entreprises à ce niveau.</li>
                            <li><strong>Médiane et Quartiles</strong>: La médiane ({round(stats['50%'], 2)}) et l'intervalle interquartile ({round(stats['25%'], 2)} à {round(stats['75%'], 2)}) montrent que 50% des entreprises ont des valeurs dans cet intervalle.</li>
                            <li><strong>Valeurs extrêmes</strong>: Les points à {int(stats['max'])} (ex. Marsa Maroc) indiquent des entreprises hors norme.</li>
                            <li><strong>Comparaison</strong>: Par rapport à {'NbPersonnesQuotidien' if var == 'NbToilettes' else 'NbToilettes'}, {var} a {'une distribution moins étirée' if var == 'NbToilettes' else 'une distribution plus étirée'} (étendue {int(stats['max'] - stats['min'])} vs {1790 if var == 'NbToilettes' else 88}).</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                # Détection des valeurs aberrantes
                Q1, Q3 = stats['25%'], stats['75%']
                IQR = Q3 - Q1
                lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)][var]
                if not outliers.empty:
                    st.markdown("**Valeurs Aberrantes**")
                    st.write(f"Entreprises avec valeurs aberrantes pour {var} : {', '.join(outliers.index.tolist())}")
                    st.write(f"Valeurs : {outliers.values.tolist()}")
                else:
                    st.markdown("**Valeurs Aberrantes** : Aucune détectée.")

    # --- 4. Analyse de Corrélation ---
    st.markdown('<div class="section-header">3. Analyse de Corrélation</div>', unsafe_allow_html=True)

    with st.container():
        if len(variables_quantitatives) > 1:
            st.markdown('<div class="subheader">Corrélation entre NbToilettes et NbPersonnesQuotidien</div>', unsafe_allow_html=True)
            corr, p_value = pearsonr(df['NbToilettes'], df['NbPersonnesQuotidien'])
            st.markdown(f"""
            <div class="table-comment">
                <h4>Résultats de la Corrélation</h4>
                <ul>
                    <li><strong>Coefficient de Pearson</strong>: {round(corr, 3)}</li>
                    <li><strong>P-valeur</strong>: {round(p_value, 3)}</li>
                    <li><strong>Interprétation</strong>: {'Corrélation significative' if p_value < 0.05 else 'Corrélation non significative'} (p {'< 0.05' if p_value < 0.05 else '≥ 0.05'}).</li>
                    <li><strong>Force</strong>: {'Forte' if abs(corr) > 0.7 else 'Modérée' if abs(corr) > 0.3 else 'Faible'} ({'positive' if corr > 0 else 'négative'}).</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Nuage de points
            fig_scatter = px.scatter(df, x='NbToilettes', y='NbPersonnesQuotidien', text=df.index,
                                    title='NbToilettes vs NbPersonnesQuotidien',
                                    color_discrete_sequence=['#1E88E5'])
            fig_scatter.update_traces(textposition='top center', marker=dict(size=12))
            fig_scatter.update_layout(showlegend=False)
            st.plotly_chart(fig_scatter, use_container_width=True)
            # Commentaire pour le nuage de points
            st.markdown(f"""
            <div class="graph-comment">
                <h4>Interprétation du Nuage de Points</h4>
                <ul>
                    <li><strong>Tendance</strong>: Ce nuage de points illustre la relation entre le nombre de toilettes et le nombre de personnes fréquentant les locaux.</li>
                    <li><strong>Corrélation</strong>: Une corrélation {'forte' if abs(corr) > 0.7 else 'modérée' if abs(corr) > 0.3 else 'faible'} et {'positive' if corr > 0 else 'négative'} est observée (coefficient = {round(corr, 3)}).</li>
                    <li><strong>Valeurs clés</strong>: Marsa Maroc (90 toilettes, 1800 personnes) est un point extrême, tandis que GLACIERES DU PORT et Mana mesine (2 toilettes, 10 personnes) sont au bas de l'échelle.</li>
                    <li><strong>Comparaison</strong>: Les entreprises avec plus de personnes (ex. Marsa Maroc, somaport) ont systématiquement plus de toilettes, confirmant la corrélation.</li>
                    <li><strong>Insight</strong>: {'La forte corrélation suggère que le nombre de toilettes est directement lié à la fréquentation.' if abs(corr) > 0.7 else 'La corrélation modérée indique une relation, mais d\'autres facteurs peuvent influencer.'}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Matrice de corrélation
            st.markdown('<div class="subheader">Matrice de Corrélation</div>', unsafe_allow_html=True)
            corr_matrix = df[variables_quantitatives].corr()
            fig_heatmap = ff.create_annotated_heatmap(
                z=corr_matrix.values, x=variables_quantitatives, y=variables_quantitatives,
                colorscale='Blues', annotation_text=corr_matrix.round(2).values
            )
            fig_heatmap.update_layout(title='Matrice de Corrélation', width=500, height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            # Commentaire pour la matrice de corrélation
            st.markdown(f"""
            <div class="graph-comment">
                <h4>Interprétation de la Matrice de Corrélation</h4>
                <ul>
                    <li><strong>Relation</strong>: Cette matrice montre la relation entre NbToilettes et NbPersonnesQuotidien.</li>
                    <li><strong>Valeur clé</strong>: La corrélation de {round(corr, 2)} indique une {'forte' if abs(corr) > 0.7 else 'modérée' if abs(corr) > 0.3 else 'faible'} relation {'positive' if corr > 0 else 'négative'}.</li>
                    <li><strong>Diagonale</strong>: Les valeurs de 1 représentent la corrélation parfaite d'une variable avec elle-même.</li>
                    <li><strong>Comparaison</strong>: La forte corrélation (proche de 1) confirme que les entreprises avec plus de personnes ont tendance à avoir plus de toilettes, comme observé dans le nuage de points.</li>
                    <li><strong>Insight</strong>: Cette relation suggère que la taille de l'entreprise (en termes de fréquentation) influence directement les infrastructures sanitaires.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">ℹ️ Pas assez de variables quantitatives pour une analyse de corrélation.</div>', unsafe_allow_html=True)

    # --- 5. Téléchargement des Résultats ---
    st.markdown('<div class="section-header">4. Téléchargement des Résultats</div>', unsafe_allow_html=True)
    csv = df.to_csv()
    st.download_button(
        label="📥 Télécharger les données transformées (CSV)",
        data=csv,
        file_name="donnees_enquete_transformees.csv",
        mime="text/csv"
    )