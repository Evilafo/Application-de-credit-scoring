import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import plotly.express as px
from zipfile import ZipFile
from sklearn.cluster import KMeans



plt.style.use('fivethirtyeight')
#sns.set()
sns.set_theme(style="darkgrid")
#sns.set_style('darkgrid')

#st.set_page_config(page_title='Dashbord de Credit Scoring - Evilafo' ,layout="wide",page_icon='📊')

def main() :

    @st.cache
    #@st.cache_resource
    def load_data():
        z = ZipFile("data/default_risk.zip")
        data = pd.read_csv(z.open('default_risk.csv'), index_col='SK_ID_CURR', encoding ='utf-8')

        z = ZipFile("data/X_sample.zip")
        sample = pd.read_csv(z.open('X_sample.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
        
        description = pd.read_csv("data/features_description.csv", 
                                  usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')

        target = data.iloc[:, -1:]

        return data, sample, target, description


    def load_model():
        '''chargement du modèle entrainé'''
        pickle_in = open('model/LGBMClassifier.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        return clf


    @st.cache(allow_output_mutation=True)
    #@st.cache_resource
    def load_knn(sample):
        knn = knn_training(sample)
        return knn

    def calculate_feature_importance(X, model):
        importances = model.feature_importances_
        feature_names = X.columns
        return pd.Series(importances, index=feature_names)


    @st.cache
    #@st.cache_resource
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                     round(data["AMT_INCOME_TOTAL"].mean(), 2),
                     round(data["AMT_CREDIT"].mean(), 2)]

        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        targets = data.TARGET.value_counts()

        return nb_credits, rev_moy, credits_moy, targets


    def identite_client(data, id):
        data_client = data[data.index == int(id)]
        return data_client

    @st.cache
    #@st.cache_resource
    def load_age_population(data):
        data_age = round((data["DAYS_BIRTH"]/365), 2)
        return data_age

    @st.cache
    #@st.cache_resource
    def load_income_population(sample):
        df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income

    @st.cache
    #@st.cache_resource
    def load_prediction(sample, id, clf):
        X=sample.iloc[:, :-1]
        score = clf.predict_proba(X[X.index == int(id)])[:,1]
        return score

    @st.cache
    #@st.cache_resource
    def load_kmeans(sample, id, mdl):
        index = sample[sample.index == int(id)].index.values
        index = index[0]
        data_client = pd.DataFrame(sample.loc[sample.index, :])
        df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
        df_neighbors = pd.concat([df_neighbors, data], axis=1)
        return df_neighbors.iloc[:,1:].sample(10)

    @st.cache
    #@st.cache_resource
    def knn_training(sample):
        knn = KMeans(n_clusters=2).fit(sample)
        return knn 



    #Loading data……
    data, sample, target, description = load_data()
    id_client = sample.index.values
    clf = load_model()


    #######################################
    # SIDEBAR
    #######################################

    #Title display
    html_temp = """
    <!--<h1 style="text-align:center">Tableau de bord Scoring Credit📈💰 </h1>-->
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
    <p style="font-size: 20px; font-weight: bold; text-align:center">Aide à la décision de crédit, Prédiction de défaut de paiement </p>
    
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown(html_temp2, unsafe_allow_html=True)
    st.markdown(html_temp3, unsafe_allow_html=True)

    #Customer ID selection
    st.sidebar.header("**Informations Generales**")

    #Loading selectbox
    chk_id = st.sidebar.selectbox("Rechercher l'ID du Client", id_client)

    #Loading general info
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)


    ### Display of information in the sidebar ###
    #Number of loans in the sample
    #st.sidebar.markdown("<u>Nombre de prêts dans l'échantillon :</u>", unsafe_allow_html=True)
    #st.sidebar.text(nb_credits)

    #Average income
    #st.sidebar.markdown("<u>Revenu moyen (USD) :</u>", unsafe_allow_html=True)
    #st.sidebar.text(rev_moy)

    #AMT CREDIT
    #st.sidebar.markdown("<u>Montant moyen du prêt (USD) :</u>", unsafe_allow_html=True)
    #st.sidebar.text(credits_moy)
    
    #PieChart
    #st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.pie(targets, explode=[0, 0.1], labels=['Solvable', 'Non solvable'], autopct='%1.1f%%', startangle=90)
    st.sidebar.pyplot(fig)

    #Copyright
    with st.sidebar:
        st.markdown("&nbsp; &nbsp; &nbsp;")
        img_gallery = """<div style="display: flex;">
        <a href="https://github.com/Evilafo"><img src="https://upload.wikimedia.org/wikipedia/commons/c/c2/GitHub_Invertocat_Logo.svg" alt="icon" height="40" style="height: 40px;  margin-right: 10px" /></a>
        <a href="https://www.linkedin.com/in/emmanuel-evilafo-838734165"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcROKs8r8Zd_xOz-qdO6Mk9bQXGh-CP4kiHqJtIsZ2CP2Q&s" alt="icon" width="40" style="width: 40px; height: 40px; margin-right: 10px; margin-bottom: 0px;" /></a>
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
        L'objectif de cette application est d'évaluer le risque de défaut de paiement d'un emprunteur potentiel en utilisant des données démographiques et financières.
        """,
    )
    
    #ID du client Sidebar
    st.write("Numéro du client sélectionné:", chk_id)




    #Affichage de la solvabilité du client
    st.header("**Analyse du dossier client**")
    prediction = load_prediction(sample, chk_id, clf)
    #Calcul probabilite
    predict = round(float(prediction)*100)

    if predict < 1 :
        message = "Très faible"
        couleur = "green "
        st.markdown(f""" Probabilité de risque de défaut : <b> :green[{predict}%] {message} </b> """, unsafe_allow_html=True)
    elif predict < 5 :
        message = "Faible"
        couleur = "green "
        st.markdown(f""" Probabilité de risque de défaut : <b> :green[{predict}%] {message} </b> """, unsafe_allow_html=True)
    elif predict < 10 :
        message = "Moyen"
        couleur = "Blue "
        st.markdown(f""" Probabilité de risque de défaut : <b> :blue[{predict}%] {message} </b> """, unsafe_allow_html=True)
    elif predict < 20 :
        message = "Elevé"
        couleur = "orange "
        st.markdown(f""" Probabilité de risque de défaut : <b> :orange[{predict}%] {message} </b> """, unsafe_allow_html=True)
    elif predict >= 20 :
        message = "Très élevé"
        couleur = "rouge"
        st.markdown(f""" Probabilité de risque de défaut : <b> :red[{predict}%] {message} </b> """, unsafe_allow_html=True)   

    st.markdown("<u>Données du client:</u>", unsafe_allow_html=True)
    idcli = identite_client(data, chk_id)
    idcli2 = idcli.copy()
    idcli2.drop('TARGET', axis=1, inplace=True)
    idcli2.insert(0, 'TARGET', idcli['TARGET'])
    #st.write(idcli2)
    st.table(idcli2)


    
    #Informations du client : Genre, Age, Statut familial, Enfants...
    st.header("**Informations du client**")
    
    with st.expander("Afficher les informations du client ?"):
    #if st.checkbox("Afficher les informations du client ?"):

        infos_client = identite_client(data, chk_id)
        code_genre = infos_client["CODE_GENDER"].values[0]
        st.markdown(f"""**Genre : ** {code_genre} """)
        #st.write.markdown(f"""**Genre : ** {code_genre} """)
        #st.write("**Genre : **", infos_client["CODE_GENDER"].values[0])
        st.write("**Age : **{:.0f} ans".format(int(infos_client["DAYS_BIRTH"]/365)))
        st.write("**Statut familial : **", infos_client["NAME_FAMILY_STATUS"].values[0])
        st.write("**Nombre d'enfant : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))

        #Age distribution plot
        data_age = load_age_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        #sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
        #ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="green", linestyle='--')
        sns.histplot(data_age, edgecolor = 'k', color="skyblue", bins=20)
        ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="red", linestyle='--')
        ax.set(title='Age du client', xlabel='Age(Années)', ylabel='')
        st.pyplot(fig)
    
        
        st.subheader("*Revenu (USD)*")
        st.write("**Revenu total : **{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
        st.write("**Montant du crédit : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
        st.write("**Annuité de crédit : **{:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
        #st.write("**Montant du bien pour crédit : **{:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))
        st.write("**Montant du bien pour pour lequel le prêt est accordé : **{:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0])) 
        
        #Diagramme de répartition des revenus
        data_income = load_income_population(data)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="goldenrod", bins=10)
        ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
        ax.set(title='Revenu du client', xlabel='Revenu (USD)', ylabel='')
        st.pyplot(fig)
        
        #Relation Âge / Revenu Total graphique interactif
        data_sk = data.reset_index(drop=False)
        data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH']/365).round(1)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig = px.scatter(data_sk, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL", 
                         size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                         hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])

        fig.update_layout({'plot_bgcolor':'#f0f0f0'}, 
                          title={'text':"Relation Âge / Revenu Total", 'x':0.5, 'xanchor': 'center'}, 
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))


        fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Revenu Total", title_font=dict(size=18, family='Verdana'))

        st.plotly_chart(fig)
    
    #else:
    #    st.markdown("<i>…</i>", unsafe_allow_html=True)

        
    #Feature importance / description // supprimé
         
    
    #Feature importance / description \\ supprimé

    #Similar customer files display
    #chk_voisins = st.checkbox("Afficher les dossiers similaires ?")
    with st.expander("Afficher les dossiers similaires ?") :
    #if chk_voisins:
        #knn = load_knn(sample)
        #st.markdown("<u>Liste des 10 dossiers les plus proches de ce Client :</u>", unsafe_allow_html=True)
        #st.dataframe(load_kmeans(sample, chk_id, knn))
        #st.markdown("<i>Target 1 = Client non solvables</i>", unsafe_allow_html=True)


        knn = load_knn(sample)
        st.markdown("<u>Liste des 10 dossiers les plus proches de ce Client :</u>", unsafe_allow_html=True)
        dossier_proche1 = load_kmeans(sample, chk_id, knn)
        dossier_proche2 = dossier_proche1.copy()
        dossier_proche2.drop('TARGET', axis=1, inplace=True)
        dossier_proche2.insert(0, 'TARGET', dossier_proche1['TARGET'])
        st.dataframe(dossier_proche2)
        st.markdown("<i>Target 1 = Clients non solvables</i>", unsafe_allow_html=True)

    


    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            .stActionButton {visibility: hidden;}
            .viewerBadge_link__qRIco {visibility: hidden;}
            .viewerBadge_container__r5tak {visibility: hidden;}
            .styles_viewerBadge__CvC9N {visibility: hidden;}
            .viewerBadge_container__r5tak {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)




    #viewerBadge_container__r5tak styles_viewerBadge__CvC9N
    





    #else:
        #st.markdown("<i>…</i>", unsafe_allow_html=True)
        
        
    #st.markdown('***')
    #st.markdown("Par Emmanuel Evilafo")
    #st.markdown("*Code from [Github](https://github.com/Evilafo)* ")



#if __name__ == '__main__':
#    main()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Dashbord de Credit Scoring - Evilafo", layout="wide",page_icon='📊')
    main()
