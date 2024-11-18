from fastapi.testclient import TestClient
from main import  app

client = TestClient(app)

def test_predict():
    csv_test = (
        "ID;review_content;review_title;review_stars;product\n"
        "1;Produit excellent;Super;5;A\n"
        "2;Produit horrible;Terrible;1;B\n"
    )

    response = client.post(
        "/predict",
        files={"file": ("test.csv", csv_test, "text/csv")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2  # deux lignes dans le fichier d'entrée
