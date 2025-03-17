from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def set_prequalified_by_main_category(df, query, threshold=0.3):
    """
    Marks vendors as prequalified if their 'main_category' matches the query
    based on TF-IDF cosine similarity.
    """
    vectorizer = TfidfVectorizer()
    combined = df['main_category'].fillna('').tolist() + [query]
    
    tfidf_matrix = vectorizer.fit_transform(combined)
    
    query_vector = tfidf_matrix[-1]
    category_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(query_vector, category_vectors).flatten()
    
    df['main_category_similarity'] = similarities
    df['prequalified'] = similarities >= threshold  # mark True if above threshold
    
    return df
