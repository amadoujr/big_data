from io import BytesIO
import pandas as pd
from fastapi import FastAPI, UploadFile, File
import pickle
from pydantic import BaseModel
from tp_review import *
import os

model_path = os.path.join(os.path.dirname(__file__), 'logistic_reg_review')

model = pickle.load(open(model_path, 'rb'))

class RequestBody(BaseModel):
    ID: int
    review_content: object
    review_title : object
    review_stars : int
    product      : object

app = FastAPI()

async def process_csv(file: UploadFile):
    # Lis le fichier
    content = await file.read()
    # Charge le CSV en DataFrame
    df = pd.read_csv(BytesIO(content), sep=";")
    # Vérifie que le fichier correspond au format de la classe RequestBody
    expected_columns = RequestBody.model_json_schema()["properties"].keys()
    if not all(col in df.columns for col in expected_columns):
        raise ValueError("Le fichier CSV doit contenir les colonnes suivantes: " + ", ".join(expected_columns))
    
    return df

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        df = await process_csv(file)
    except ValueError as e:
        return {"error": str(e)}
    
    # Étape 1 : Prétraitement initial
    df_cleaned = pre_process_data(df)
    # Étape 2 : Extraction des mots et caractères dans les revues
    df_cleaned = extract_words_char_in_review(df_cleaned)
    # Étape 3 : Comptage des mots-clés
    df_cleaned['keyword_count'] = df_cleaned['review_content'].apply(count_keywords)
    # Prediction!
    predictions = model.predict(df_cleaned[['content_nb_char', 'content_nb_words','review_stars']])
    # Retourne les résultats
    return {
        "predictions": predictions.tolist(),
    }