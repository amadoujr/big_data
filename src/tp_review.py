import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


## Chargement et pré-traitement des données

def pre_process_data(data):
    """
    cette fonction va charger le fichier au format csv,
    retirer les review dupliqués, puis enfin retirer la colonne
    product.
    """
    ## suppression de la colonne product
    dataset_cleaned = data.drop_duplicates(subset=['review_content'])
    dataset_cleaned = dataset_cleaned.drop(columns=['product'])
    return dataset_cleaned

## Extraction des features syntaxiques

# Fonction pour extraire (nombre de caractères/mots) dans une review
def extract_words_char_in_review(df):
    df['content_nb_char'] = df['review_content'].apply(len)
    df['content_nb_words'] = df['review_content'].apply(lambda x : len(x.split()))
    return df


keywords = ['horrible','pas bon','pas bien','bien','bon', 'excellent', 'parfait', 'déçu', 'mauvais','pas mauvais']
# Fonction pour détecter des mots-clés dans le contenu
def count_keywords(text):
    detected_keywords = [word for word in keywords if word in text.lower()]
    return detected_keywords

# Fonction pour ajouter une colonne binaire (sentiment) pour 
# les avis positifs (3-5 étoiles) et négatifs (1-2 étoiles)  

def add_sentiment(df,column='review_stars'):
    df['sentiment'] = df[column].apply(lambda x : 'positive' if x >= 3 else 'negative')
    return df 


def train_model(model,X,y):
    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)
    model.fit(X_train,y_train)

    # Prédiction sur le set de test
    y_pred = model.predict(X_test)

    # Évaluation du modèle
    print(classification_report(y_test, y_pred))
    