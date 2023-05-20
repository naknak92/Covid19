# Analyse de la situation COVID-19 en Europe

Ce code est une application Streamlit qui permet d'analyser la situation de la COVID-19 en Europe en utilisant différentes visualisations et analyses de données. Voici un aperçu des différentes fonctionnalités disponibles :

## Installation

Pour exécuter ce code, vous devez installer les bibliothèques suivantes :

- streamlit
- pandas
- numpy
- datetime
- scipy
- matplotlib
- plotly
- scikit-learn

Vous pouvez les installer en utilisant pip :

pip install streamlit pandas numpy datetime scipy matplotlib plotly scikit-learn


## Utilisation

Une fois les bibliothèques installées, vous pouvez exécuter le code en utilisant la commande suivante :


streamlit run nom_du_fichier.py


L'application Streamlit s'ouvrira dans votre navigateur, où vous pourrez interagir avec les différentes fonctionnalités.

## Fonctionnalités

L'application propose les fonctionnalités suivantes :

1. **Top 10 pays avec le plus de cas de COVID-19 en Europe** : Affiche un graphique à barres des pays avec le plus grand nombre de cas de COVID-19 en Europe.
2. **Top 10 pays avec le plus de décès de COVID-19 en Europe** : Affiche un graphique à barres des pays avec le plus grand nombre de décès dus à la COVID-19 en Europe.
3. **Régression linéaire du nombre de vaccinations en France par rapport aux cas totaux de COVID-19** : Effectue une régression linéaire pour prédire le nombre de personnes vaccinées en France en fonction du nombre total de cas de COVID-19, et affiche le graphique de la régression linéaire.
4. **Top 10 pays avec le plus de cas de grippe dans le monde** : Affiche un graphique à barres des pays avec le plus grand nombre de cas de grippe chez les personnes de plus de 65 ans.
5. **Prédiction du nombre de personnes vaccinées dans le monde le 31 décembre 2023** : Utilise un modèle de croissance logistique pour prédire le nombre de personnes vaccinées dans le monde d'ici le 31 décembre 2023, et affiche le résultat avec un graphique.
6. **Afficher tous les graphiques** : Affiche tous les graphiques disponibles en une seule fois.

Vous pouvez sélectionner l'une de ces options dans le menu déroulant de l'application Streamlit pour visualiser les données correspondantes.

**Note** : Certaines fonctionnalités nécessitent le téléchargement de fichiers de données supplémentaires à partir de sources en ligne.

N'hésitez pas à explorer les différentes fonctionnalités de l'application pour obtenir des informations sur la situation de la COVID-19 en Europe.
