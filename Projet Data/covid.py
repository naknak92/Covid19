import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Titre de l'application
st.title("Analyse de la situation COVID-19 en Europe")

# Récupérer les données de COVID-19 (remplacez cette ligne par la méthode que vous utilisez pour obtenir les données)
data_covid = pd.read_csv("europe_covid.csv")

# Récupérer les données de grippe
data_grippe = pd.read_csv("grippe2.csv", delimiter=",")

# Options du menu déroulant
options = ['Top 10 pays avec le plus de cas de COVID-19 en Europe', 
           'Top 10 pays avec le plus de décès de COVID-19 en Europe', 
           'Régression linéaire du nombre de vaccinations en France par rapport aux cas totaux de COVID-19',
           'Top 10 pays avec le plus de cas de grippe dans le monde',
           'Prédiction du nombre de personnes vaccinées dans le monde le 31 décembre 2023',
           'Nombre de mort du Covid par pays en Europe',
           'Afficher tous les graphiques']

# Sélection de l'option dans le menu déroulant
selected_option = st.selectbox("Sélectionnez une option", options)

if selected_option == options[0]:
    # Trier les données par cas et prendre les 10 premiers pays
    df_cases = data_covid.sort_values(by=['Total Cases'], ascending=False).reset_index().head(10)

    # Créer le graphique des cas
    fig_cases = px.bar(df_cases, x='Country/Other', y='Total Cases', color='Total Cases',
                       color_continuous_scale='reds')

    fig_cases.update_layout(title='Top 10 pays avec le plus de cas de COVID-19 en Europe',
                            title_x=0.5,
                            title_font=dict(size=16, color='DarkRed'))

    # Afficher le graphique des cas dans l'application Streamlit
    st.plotly_chart(fig_cases)

if selected_option == options[1]:
    # Trier les données par décès et prendre les 10 premiers pays
    df_deaths = data_covid.sort_values(by=['Total Deaths'], ascending=False).reset_index().head(10)

    # Créer le graphique des décès
    fig_deaths = px.bar(df_deaths, x='Country/Other',
                        y='Total Deaths',
                        color='Total Deaths',
                        color_continuous_scale='gray')

    fig_deaths.update_layout(title='Top 10 pays avec le plus de décès de COVID-19 en Europe',
                             title_x=0.5,
                             title_font=dict(size=16, color='Black'))

    # Afficher le graphique des décès dans l'application Streamlit
    st.plotly_chart(fig_deaths)

if selected_option == options[2]:
    # Chargement des données de cas de COVID-19 en France
    data_cases = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')
    data_cases = data_cases[data_cases['location']=='France'][['date', 'total_cases']]
    data_cases = data_cases.dropna()

    # Chargement des données de vaccination COVID-19 en France
    data_vaccinations = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/country_data/France.csv')
    data_vaccinations = data_vaccinations[['date', 'people_vaccinated']]
    data_vaccinations = data_vaccinations.dropna()

    # Joindre les données de cas et de vaccination en utilisant la date comme clé
    data = pd.merge(data_cases, data_vaccinations, on='date')

    # Préparation des données pour la régression linéaire
    X = data[['total_cases']]
    y = data['people_vaccinated']

    # Création et entraînement du modèle de régression linéaire
    reg = LinearRegression()
    reg.fit(X, y)

    # Prédiction du nombre de personnes vaccinées
    predicted_vaccinations = reg.predict(X)

    # Création du graphique de la régression linéaire
    fig_regression = go.Figure()
    fig_regression.add_trace(go.Scatter(x=X['total_cases'], y=y, mode='markers', name='Données réelles'))
    fig_regression.add_trace(go.Scatter(x=X['total_cases'], y=predicted_vaccinations, mode='lines', name='Régression linéaire'))

    fig_regression.update_layout(title='Régression linéaire du nombre de vaccinations en France par rapport aux cas totaux de COVID-19',
                                 xaxis_title='Nombre total de cas de COVID-19',
                                 yaxis_title='Nombre total de personnes vaccinées',
                                 title_font=dict(size=16),
                                 legend=dict(x=0.7, y=0.9))

    # Afficher le graphique de la régression linéaire dans l'application Streamlit
    st.plotly_chart(fig_regression)

if selected_option == options[3]:
    # Trier les données par taux de grippe chez les personnes de plus de 65 ans et prendre les 10 premiers pays
    df_grippe = data_grippe.sort_values(by='rate over65', ascending=False).reset_index().head(10)

    # Créer le graphique de la grippe
    fig_grippe = px.bar(df_grippe, x='Entity', y='rate over65', color='rate over65',
                        color_continuous_scale='reds')

    fig_grippe.update_layout(title='Top 10 pays avec le plus de cas de grippe',
                             title_x=0.5,
                             title_font=dict(size=16, color='DarkRed'))

    # Afficher le graphique de la grippe dans l'application Streamlit
    st.plotly_chart(fig_grippe)

if selected_option == options[4]:
    # Lire les données du fichier CSV
    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
    data = pd.read_csv(url)

    # Convertir les dates en format datetime
    data["date"] = pd.to_datetime(data["date"])

    # Filtrer les données jusqu'à 2021
    data = data[data["date"] < datetime(2022, 1, 1)]

    # Aggréger les données mondiales
    world_data = data.groupby("date").agg({"people_vaccinated": "sum"}).reset_index()

    # Préparer les données pour la régression
    world_data["days_since_start"] = (world_data["date"] - world_data["date"].min()).dt.days
    X = world_data["days_since_start"].values
    y = world_data["people_vaccinated"].values

    # Fonction de croissance logistique
    def logistic_model(x, a, b, c):
        return c / (1 + np.exp(-(x - b) / a))

    # Estimer les paramètres du modèle de croissance loagistique
    initial_parameters = [2, 100, 2000]
    params, _ = curve_fit(logistic_model, X, y, p0=initial_parameters)

    # Faire des prédictions pour la fin de 2023
    future_date = datetime(2023, 12, 31)
    days_since_start_2023 = (future_date - world_data["date"].min()).days
    X_future = np.array([days_since_start_2023])
    y_future_pred = logistic_model(X_future, *params)

    number_vaccinated = y_future_pred[0]
    formatted_number = f"{number_vaccinated / 1e9:.2f} Milliard" if number_vaccinated >= 1e9 else f"{number_vaccinated / 1e6:.2f} million"

    # Afficher la prédiction dans l'application Streamlit
    st.write("Nombre de personnes vaccinées dans le monde prévu d'ici le 31 décembre 2023:", formatted_number)

    # Visualiser les données et les prédictions
    plt.figure(figsize=(12, 6))
    plt.scatter(X, y, label="Données")
    plt.plot(np.arange(max(X) + 100), logistic_model(np.arange(max(X) + 100), *params), 'r-', label="Modèle logistique ajusté")
    plt.axvline(X_future, linestyle="--", color="gray", label="Fin de 2023")
    plt.scatter(X_future, y_future_pred, marker="o", color="red", label="Prédiction pour fin de 2023")
    plt.xlabel("Jours depuis le début")
    plt.ylabel("Nombre de personnes vaccinées")
    plt.legend()

    # Afficher le graphique dans l'application Streamlit
    st.pyplot(plt)


if selected_option == options[5]:
    #afficher le nombre de mort par pays en europe
    data2 = pd.read_csv("europe_covid.csv", delimiter=",")

    df2 = data2.sort_values(by=['Deaths/ 1M pop'], ascending=False).reset_index()

    fig = px.choropleth(df2,
                    locations='Country/Other',
                    locationmode='country names',
                    color='Total Deaths',
                    scope='europe',
                    hover_name='Country/Other',
                    color_continuous_scale='reds')

    fig.update_layout(title='COVID-19 Deaths in European Countries',
                  title_x=0.5,
                  title_font=dict(size=16, color='Darkred'),
                  geo=dict(showframe=False,
                           showcoastlines=False,
                           projection_type='equirectangular'))

    st.plotly_chart(fig)

if selected_option == options[6]:
    # Trier les données par cas et prendre les 10 premiers pays
    df_cases = data_covid.sort_values(by=['Total Cases'], ascending=False).reset_index().head(10)

    # Créer le graphique des cas
    fig_cases = px.bar(df_cases, x='Country/Other', y='Total Cases', color='Total Cases',
                       color_continuous_scale='reds')

    fig_cases.update_layout(title='Top 10 pays avec le plus de cas de COVID-19 en Europe',
                            title_x=0.5,
                            title_font=dict(size=16, color='DarkRed'))

    # Trier les données par décès et prendre les 10 premiers pays
    df_deaths = data_covid.sort_values(by=['Total Deaths'], ascending=False).reset_index().head(10)

    # Créer le graphique des décès
    fig_deaths = px.bar(df_deaths, x='Country/Other',
                        y='Total Deaths',
                        color='Total Deaths',
                        color_continuous_scale='gray')

    fig_deaths.update_layout(title='Top 10 pays avec le plus de décès de COVID-19 en Europe',
                             title_x=0.5,
                             title_font=dict(size=16, color='Black'))

    # Chargement des données de cas de COVID-19 en France
    data_cases = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')
    data_cases = data_cases[data_cases['location']=='France'][['date', 'total_cases']]
    data_cases = data_cases.dropna()

    # Chargement des données de vaccination COVID-19 en France
    data_vaccinations = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/country_data/France.csv')
    data_vaccinations = data_vaccinations[['date', 'people_vaccinated']]
    data_vaccinations = data_vaccinations.dropna()

    # Joindre les données de cas et de vaccination en utilisant la date comme clé
    data = pd.merge(data_cases, data_vaccinations, on='date')

    # Préparation des données pour la régression linéaire
    X = data[['total_cases']]
    y = data['people_vaccinated']

    # Création et entraînement du modèle de régression linéaire
    reg = LinearRegression()
    reg.fit(X, y)

    # Prédiction du nombre de personnes vaccinées
    predicted_vaccinations = reg.predict(X)

    # Création du graphique de la régression linéaire
    fig_regression = go.Figure()
    fig_regression.add_trace(go.Scatter(x=X['total_cases'], y=y, mode='markers', name='Données réelles'))
    fig_regression.add_trace(go.Scatter(x=X['total_cases'], y=predicted_vaccinations, mode='lines', name='Régression linéaire'))

    fig_regression.update_layout(title='Régression linéaire du nombre de vaccinations en France par rapport aux cas totaux de COVID-19',
                                 xaxis_title='Nombre total de cas de COVID-19',
                                 yaxis_title='Nombre total de personnes vaccinées',
                                 title_font=dict(size=16),
                                 legend=dict(x=0.7, y=0.9))

    # Trier les données par taux de grippe chez les personnes de plus de 65 ans et prendre les 10 premiers pays
    df_grippe = data_grippe.sort_values(by='rate over65', ascending=False).reset_index().head(10)

    # Créer le graphique de la grippe
    fig_grippe = px.bar(df_grippe, x='Entity', y='rate over65', color='rate over65',
                        color_continuous_scale='reds')

    fig_grippe.update_layout(title='Top 10 pays avec le plus de cas de grippe en Europe',
                             title_x=0.5,
                             title_font=dict(size=16, color='DarkRed'))

    

    # Lire les données du fichier CSV
    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
    data = pd.read_csv(url)

    # Convertir les dates en format datetime
    data["date"] = pd.to_datetime(data["date"])

    # Filtrer les données jusqu'à 2021
    data = data[data["date"] < datetime(2022, 1, 1)]

    # Aggréger les données mondiales
    world_data = data.groupby("date").agg({"people_vaccinated": "sum"}).reset_index()

    # Préparer les données pour la régression
    world_data["days_since_start"] = (world_data["date"] - world_data["date"].min()).dt.days
    X = world_data["days_since_start"].values
    y = world_data["people_vaccinated"].values


    

    # Fonction de croissance logistique
    def logistic_model(x, a, b, c):
        return c / (1 + np.exp(-(x - b) / a))

    # Estimer les paramètres du modèle de croissance loagistique
    initial_parameters = [2, 100, 2000]
    params, _ = curve_fit(logistic_model, X, y, p0=initial_parameters)

    # Faire des prédictions pour la fin de 2023
    future_date = datetime(2023, 12, 31)
    days_since_start_2023 = (future_date - world_data["date"].min()).days
    X_future = np.array([days_since_start_2023])
    y_future_pred = logistic_model(X_future, *params)

    number_vaccinated = y_future_pred[0]
    formatted_number = f"{number_vaccinated / 1e9:.2f} billion" if number_vaccinated >= 1e9 else f"{number_vaccinated / 1e6:.2f} million"

    # Créer le graphique des cas
    fig_cases_all = px.bar(df_cases, x='Country/Other', y='Total Cases', color='Total Cases',
                           color_continuous_scale='reds')

    fig_cases_all.update_layout(title='Top 10 pays avec le plus de cas de COVID-19 en Europe',
                                title_x=0.5,
                                title_font=dict(size=16, color='DarkRed'))

    # Créer le graphique des décès
    fig_deaths_all = px.bar(df_deaths, x='Country/Other',
                            y='Total Deaths',
                            color='Total Deaths',
                            color_continuous_scale='gray')

    fig_deaths_all.update_layout(title='Top 10 pays avec le plus de décès de COVID-19 en Europe',
                                 title_x=0.5,
                                 title_font=dict(size=16, color='Black'))

    # Créer le graphique de la régression linéaire
    

    # Créer le graphique de la grippe
    fig_grippe_all = px.bar(df_grippe, x='Entity', y='rate over65', color='rate over65',
                            color_continuous_scale='reds')

    fig_grippe_all.update_layout(title='Top 10 pays avec le plus de cas de grippe en Europe',
                                 title_x=0.5,
                                 title_font=dict(size=16, color='DarkRed'))

    # Créer le graphique de la prédiction
    fig_prediction_all = plt.figure(figsize=(12, 6))
    plt.scatter(X, y, label="Données")
    plt.plot(np.arange(max(X) + 100), logistic_model(np.arange(max(X) + 100), *params), 'r-', label="Modèle logistique ajusté")
    plt.axvline(X_future, linestyle="--", color="gray", label="Fin de 2023")
    plt.scatter(X_future, y_future_pred, marker="o", color="red", label="Prédiction pour fin de 2023")
    plt.xlabel("Jours depuis le début")
    plt.ylabel("Nombre de personnes vaccinées")
    plt.legend()

    # Afficher les graphiques dans l'application Streamlit
    st.plotly_chart(fig_cases_all)
    st.plotly_chart(fig_deaths_all)
    st.plotly_chart(fig_grippe_all)
    st.pyplot(fig_prediction_all)

     #afficher le nombre de mort par pays en europe
    data2 = pd.read_csv("europe_covid.csv", delimiter=",")

    df2 = data2.sort_values(by=['Deaths/ 1M pop'], ascending=False).reset_index()

    fig = px.choropleth(df2,
                    locations='Country/Other',
                    locationmode='country names',
                    color='Total Deaths',
                    scope='europe',
                    hover_name='Country/Other',
                    color_continuous_scale='reds')

    fig.update_layout(title='COVID-19 Deaths in European Countries',
                  title_x=0.5,
                  title_font=dict(size=16, color='Darkred'),
                  geo=dict(showframe=False,
                           showcoastlines=False,
                           projection_type='equirectangular'))

    st.plotly_chart(fig)
