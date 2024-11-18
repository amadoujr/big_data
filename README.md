## Sentiment Analysis API

Ce projet fournit une API pour analyser les sentiments des avis clients. L'API utilise un modèle de régression logistique pour prédire si un avis est **pertinent** (1) ou **non-pertinent**(0).

### Structure du Projet

- main.py : Fichier principal contenant l'API FastAPI.
- tp_review.py : Fonctions utilitaires pour le prétraitement des données.
- train_model.py : Script pour entraîner le modèle à partir d'un fichier CSV.
- logistic_reg_review : Fichier contenant le modèle entraîné (par défaut).
- tests/api_test.py : Tests unitaires pour l'API.

### Prérequis

**Python** : Version 3.12.6
**Packages requis** : Listés dans le fichier requirements.txt

### Installation

- clonez le projet via l'url ou unzip le dossier 
- creer un environnement virtuel (pas obligatoire mais vivement conseillé)
- installez les dépendances presentes dans le fichier requirement.txt via la commande :
``` pip install -r requirements.txt ```

### Entrainement du modéle :

- Placez votre fichier d'entraînement (CSV) dans le repertoire **/data**.
- Se placez à la racine du projet et lancez le script train_model.py via la commande :
```python src/train_model.py <path_to_csv> <model_output_path>```

Voici un exemple d'exécution:
``` python src/train_model.py data/train_data.csv src/logistic_reg_review ```

### Lancement de l'API
A la racine du projet, taper la commande suivante en ligne de commande: ``` uvicorn src.main:app --reload ```

l'API sera accéssible à l'adresse suivante : http://127.0.0.1:8000

Je n'ai pas utilisé streamlit pour l'interface graphique donc il faut vous rendre sur http://127.0.0.1:8000/docs et utiliser le swagger UI puis téléchargez un fichier CSV avec la structure correcte pour prédire les sentiments.


### Tests Unitaires :

saiisir la commmande suivante (toujours en étant à la racine du projet):

``` pytest tests/api_test.py ``` 

**PS:** 
Pour utiliser un fichier modèle différent, spécifiez son nom (*model_output_path*) au moment de l'entraînement via la ligne de commande et mettez à jour **model_path** dans main.py.

