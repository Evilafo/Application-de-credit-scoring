import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
from zipfile import ZipFile
from sklearn.cluster import KMeans

plt.style.use('fivethirtyeight')
sns.set_theme(style="darkgrid")

def main():
    # Mise à jour des décorateurs de cache
    @st.cache_data
    def load_data():
        z = ZipFile("data/default_risk.zip")
        data = pd.read_csv(z.open('default_risk.csv'), index_col='SK_ID_CURR', encoding='utf-8')
        z = ZipFile("data/X_sample.zip")
        sample = pd.read_csv(z.open('X_sample.csv'), index_col='SK_ID_CURR', encoding='utf-8')
        description = pd.read_csv("data/features_description.csv", 
                                  usecols=['Row', 'Description'], index_col=0, encoding='unicode_escape')
        target = data.iloc[:, -1:]
        return data, sample, target, description

    @st.cache_resource
    def load_model():
        with open('model/LGBMClassifier.pkl', 'rb') as f:
            clf = pickle.load(f)
        return clf

    @st.cache_resource
    def knn_training(sample):
        return KMeans(n_clusters=2).fit(sample)

    @st.cache_data
    def load_kmeans(sample, id, _knn):
        data_client = sample.loc[[int(id)]]
        neighbors = _knn.predict(data_client)
        df_neighbors = pd.DataFrame(neighbors, index=data_client.index)
        df_neighbors = pd.concat([df_neighbors, sample], axis=1)
        return df_neighbors.iloc[:, 1:].sample(10)

    @st.cache_data
    def load_infos_gen(data):
        lst_infos = [
            data.shape[0],
            round(data["AMT_INCOME_TOTAL"].mean(), 2),
            round(data["AMT_CREDIT"].mean(), 2)
        ]
        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]
        targets = data.TARGET.value_counts()
        return nb_credits, rev_moy, credits_moy, targets

    @st.cache_data
    def load_age_population(data):
        return round((data["DAYS_BIRTH"] / 365), 2)

    @st.cache_data
    def load_income_population(sample):
        df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
        return df_income[df_income['AMT_INCOME_TOTAL'] < 200000]

    @st.cache_data
    def load_prediction(sample, id, _clf):
        X = sample.iloc[:, :-1]
        score = _clf.predict_proba(X.loc[[int(id)]])[:, 1]
        return score

    # Chargement initial des données
    data, sample, target, description = load_data()
    id_client = sample.index.values
    clf = load_model()

    # Initialisation du modèle KNN
    if 'knn' not in st.session_state:
        st.session_state.knn = knn_training(sample)

    #######################################
    # SIDEBAR
    #######################################
    html_temp = """
    <h1 style="text-align:center">Tableau de bord Scoring Credit📈</h1>
    """
    html_temp2 = """
    <style>
    .css-selector2 {
    background: linear-gradient(to left, #ff0000, #fffa00, #ff0000, #fffa00);
    background-size: 300% 300%;
    animation: anim 6s ease infinite;
    height:5px;
    border-radius: 50px;
    }
    @keyframes anim {
        0%{background-position:50% 50%}
        50%{background-position:100% 50%}
        100%{background-position:50% 50%}
    }
    </style>
    """
    html_temp3 = """
    <p class="css-selector2"></p>
    <p style="font-size: 20px; font-weight: bold; text-align:center">
    Aide à la décision de crédit, Prédiction de défaut de paiement </p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown(html_temp2, unsafe_allow_html=True)
    st.markdown(html_temp3, unsafe_allow_html=True)

    # Customer ID selection
    st.sidebar.header("**Informations Generales**")
    chk_id = st.sidebar.selectbox("Rechercher l'ID du Client", id_client)

    # Loading general info
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.pie(targets, explode=[0, 0.1], labels=['Solvable', 'Non solvable'], autopct='%1.1f%%', startangle=90)
    st.sidebar.pyplot(fig)

    # Copyright
    with st.sidebar:
        st.markdown("&nbsp; &nbsp; &nbsp;")
        img_gallery = """<div style="display: flex;">
        <a href="https://github.com/Evilafo"><img  src="https://upload.wikimedia.org/wikipedia/commons/c/c2/GitHub_Invertocat_Logo.svg"  alt="icon" height="40" style="height: 40px;  margin-right: 10px" /></a>
        <a href="https://www.linkedin.com/in/emmanuel-evilafo-838734165"><img  src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcROKs8r8Zd_xOz-qdO6Mk9bQXGh-CP4kiHqJtIsZ2CP2Q&s" alt="icon" width="40" style="width: 40px; height: 40px; margin-right: 10px; margin-bottom: 0px;" /></a>
        <a href="https://www.kaggle.com/emmanuelevilafo"><img src="https://www.kaggle.com/static/images/site-logo.svg" alt="icon" height="40" style="height: 40px; margin-right: 0px; margin-bottom: 0px;" /></a></div>"""
        st.markdown(img_gallery, unsafe_allow_html=True)
        st.markdown("&nbsp;")
        st.caption("© Made by Evilafo 2023. All rights reserved.")
        st.markdown(
            '<h6>By <a href="https://www.linkedin.com/in/emmanuel-evilafo-838734165">Evilafo</a></h6>',
            unsafe_allow_html=True,
        )

    ####################################### 
    # PAGE D'ACCUEIL - CONTENU PRINCIPAL
    #######################################
    st.success(
        """
        L'objectif de cette application est d'évaluer le risque de défaut de paiement d'un emprunteur potentiel en utilisant des données démographiques et financières. Le seuil est de 10%.
        """,
    )
    st.write("Numéro du client sélectionné:", chk_id)

    # Affichage de la solvabilité du client
    st.header("**Analyse du dossier client**")
    prediction = load_prediction(sample, chk_id, clf)
    predict = round(float(prediction) * 100)
    decisionsolvable = "(Solvable)"
    decisionnonsolvable = "(Non solvable)"

    if predict < 1:
        message, couleur, statut = "Très faible", "green", decisionsolvable
    elif predict < 5:
        message, couleur, statut = "Faible", "green", decisionsolvable
    elif predict < 10:
        message, couleur, statut = "Moyen", "blue", decisionsolvable
    elif predict < 20:
        message, couleur, statut = "Elevé", "orange", decisionnonsolvable
    else:
        message, couleur, statut = "Très élevé", "red", decisionnonsolvable

    st.markdown(f"""
        Probabilité de risque de défaut : <b> :{couleur}[{predict}%] {message} :{couleur}[{statut}] </b>
    """, unsafe_allow_html=True)

    # Données du client
    st.markdown("<u>Données du client:</u>", unsafe_allow_html=True)
    idcli = data.loc[[int(chk_id)]]
    idcli2 = idcli.copy()
    idcli2.insert(0, 'TARGET', idcli.pop('TARGET'))
    st.table(idcli2)

    # Informations détaillées du client
    st.header("**Informations du client**")
    with st.expander("Afficher les informations du client ?"):
        infos_client = idcli
        st.markdown(f"**Genre :** {infos_client['CODE_GENDER'].values[0]}")
        st.markdown(f"**Age :** {int(infos_client['DAYS_BIRTH'] / 365)} ans")
        st.markdown(f"**Statut familial :** {infos_client['NAME_FAMILY_STATUS'].values[0]}")
        st.markdown(f"**Nombre d'enfant :** {int(infos_client['CNT_CHILDREN'].values[0])}")

        # Graphique de distribution de l'âge
        data_age = load_age_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor='k', color="skyblue", bins=20)
        ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="red", linestyle='--')
        ax.set(title='Age du client', xlabel='Age (Années)', ylabel='')
        st.pyplot(fig)

        # Revenus du client
        st.subheader("*Revenu (USD)*")
        st.markdown(f"**Revenu total :** {infos_client['AMT_INCOME_TOTAL'].values[0]:.0f}")
        st.markdown(f"**Montant du crédit :** {infos_client['AMT_CREDIT'].values[0]:.0f}")
        st.markdown(f"**Annuité de crédit :** {infos_client['AMT_ANNUITY'].values[0]:.0f}")
        st.markdown(f"**Montant du bien pour lequel le prêt est accordé :** {infos_client['AMT_GOODS_PRICE'].values[0]:.0f}")

        # Diagramme de répartition des revenus
        data_income = load_income_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor='k', color="goldenrod", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title='Revenu du client', xlabel='Revenu (USD)', ylabel='')
        st.pyplot(fig)

        # Relation Âge / Revenu Total graphique interactif
        data_sk = data.reset_index(drop=False)
        data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH'] / 365).round(1)
        fig = px.scatter(data_sk, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL", 
                         size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                         hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])
        fig.update_layout({'plot_bgcolor': '#f0f0f0'}, 
                          title={'text': "Relation Âge / Revenu Total", 'x': 0.5, 'xanchor': 'center'}, 
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))
        fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Revenu Total", title_font=dict(size=18, family='Verdana'))
        st.plotly_chart(fig)

    # Dossiers similaires
    with st.expander("Afficher les dossiers similaires ?"):
        st.markdown("<u>Liste des 10 dossiers les plus proches de ce Client :</u>", unsafe_allow_html=True)
        dossier_proche = load_kmeans(sample, chk_id, st.session_state.knn)
        dossier_proche2 = dossier_proche.copy()
        dossier_proche2.insert(0, 'TARGET', dossier_proche.pop('TARGET'))
        st.dataframe(dossier_proche2)
        st.markdown("<i>Target 1 = Clients non solvables</i>", unsafe_allow_html=True)

    # Masquage des éléments Streamlit par défaut
    hide_streamlit_style = """
    <style>
    ._profilePreview_gzau3_63, ._link_gzau3_10, #MainMenu, .stActionButton,
    .viewerBadge_link__qRIco, .viewerBadge_container__r5tak, .styles_viewerBadge__CvC9N {
        display: none; visibility: hidden;
    }
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Dashbord de Credit Scoring - Evilafo", layout="wide", page_icon='📊'
    )
    main()
