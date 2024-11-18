# train_model.py
import argparse
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from tp_review import pre_process_data, extract_words_char_in_review, add_sentiment

def main(input_csv, model_output):
    """
    Cette fonction permet de lire un fichier csv,
    d'appliquer un modéle de LogisticRegression sur nos données
    """
    df = pd.read_csv(input_csv, sep=";")
    # Prétraite les données
    df_cleaned = pre_process_data(df)
    df_cleaned = extract_words_char_in_review(df_cleaned)
    df_cleaned = add_sentiment(df_cleaned, 'review_stars')
    
    # Features et labels
    X = df_cleaned[['content_nb_char', 'content_nb_words', 'review_stars']]
    y = df_cleaned['sentiment']

    # entrainement du modéle
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Sauvegarde le modèle
    with open(model_output, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modèle entraîné et sauvegardé dans {model_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne un modèle à partir d'un fichier CSV")
    parser.add_argument("input_csv", type=str, help="Chemin vers le fichier CSV d'entraînement")
    parser.add_argument("model_output", type=str, help="Chemin pour sauvegarder le modèle entraîné")
    args = parser.parse_args()
    main(args.input_csv, args.model_output)
