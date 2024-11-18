import pytest
import pandas as pd
from tp_review import pre_process_data, extract_words_char_in_review, add_sentiment, count_keywords

@pytest.fixture
def load_data():
    # Fixture pour générer des données de test
    data = {
        'review_content': ["Produit excellent et bon", "Produit horrible"],
        'review_stars': [5, 1],
        'product': ['A', 'B']
    }
    return pd.DataFrame(data)

def test_pre_process_data(load_data):
    df_processed = pre_process_data(load_data)
    assert 'product' not in df_processed.columns
    assert len(df_processed) == len(load_data.drop_duplicates(subset=['review_content']))

def test_extract_words_char_in_review(load_data):
    df_with_features = extract_words_char_in_review(load_data)
    assert 'content_nb_char' in df_with_features.columns
    assert 'content_nb_words' in df_with_features.columns

def test_count_keywords():
    text = "Produit excellent et pas bon"
    detected_keywords = count_keywords(text)
    assert 'excellent' in detected_keywords
    assert 'pas bon' in detected_keywords

def test_add_sentiment(load_data):
    df_with_sentiment = add_sentiment(load_data)
    assert 'sentiment' in df_with_sentiment.columns
    assert df_with_sentiment['sentiment'][0] == 'positive'
    assert df_with_sentiment['sentiment'][1] == 'negative'
