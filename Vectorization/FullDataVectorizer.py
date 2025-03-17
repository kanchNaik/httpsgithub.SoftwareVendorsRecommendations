from sklearn.feature_extraction.text import TfidfVectorizer
import json
from Vectorization.VectorizerUtility import save_vectorizer
from CommonProcessingUtility import load_nltk_data, load_data, preprocess_text,\
      load_stemmer_lemmatizer_stopwords, clean_json_for_csv
from MongoUtility import vectorize_data_mongo

def generate_tfidf_per_row_withSave(df_tfidf, directory_path):
    # Initialize the new columns in the DataFrame to store vectors and paths
    df_tfidf['vectors'] = None
    df_tfidf['vectorizer_paths'] = None
    lemmatizer, stemmer, stop_words = load_stemmer_lemmatizer_stopwords()

    for idx, row in df_tfidf.iterrows():
        vectors = {}  # This will store the actual vectors
        vectorizer_paths = {}  # This will store the vectorizer paths
        column_value_as_string = str(row['Features'])

        # Check if the string is empty or just whitespace
        if column_value_as_string.strip():
            try:
                featuresValues = json.loads(column_value_as_string)

                # Iterate over features and generate TF-IDF vectors
                for categories in featuresValues:
                    for feature in categories.get("features", []):
                        name = feature.get('name', '')
                        description = feature.get('description', '')
                        description += ' ' + feature.get('name', '') + ' ' + categories.get('Category', '') +  ' ' + row['main_category']
                        
                        if description.strip():
                            processed_description = preprocess_text(stemmer, lemmatizer, stop_words, description)
                            vectorizer = TfidfVectorizer()
                            tfidf_matrix = vectorizer.fit_transform([processed_description])
                            vectorizer_path = save_vectorizer(vectorizer, idx, name, directory_path)
                            
                            # Store the vector and path separately
                            vectors[name] = tfidf_matrix.toarray().tolist()
                            vectorizer_paths[name] = vectorizer_path
                        else:
                            vectors[name] = None  # Add None if description is empty
                            vectorizer_paths[name] = None  # Add None if description is empty

                # Store the vectors and vectorizer paths in separate columns
                df_tfidf.at[idx, 'vectors'] = clean_json_for_csv(json.dumps(vectors))  # Store vectors
                df_tfidf.at[idx, 'vectorizer_paths'] = clean_json_for_csv(json.dumps(vectorizer_paths))  # Store paths

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for row {idx}: {e}")
                df_tfidf.at[idx, 'vectors'] = None  # Store None on JSON error
                df_tfidf.at[idx, 'vectorizer_paths'] = None  # Store None for vectorizer path on JSON error
        else:
            print(f"Empty or invalid JSON for row {idx}")
            df_tfidf.at[idx, 'vectors'] = None  # Store None for empty rows
            df_tfidf.at[idx, 'vectorizer_paths'] = None  # Store None for empty rows

    return df_tfidf

def generate_tfidf_per_row(df_tfidf, directory_path=None):
    """
    Generates TF-IDF vectors for each row in the DataFrame based on text features
    provided in a JSON-formatted 'Features' column.

    Args:
        df_tfidf (pd.DataFrame): DataFrame containing a 'Features' column with JSON strings
            describing various features and associated text data.
        directory_path (str, optional): Unused parameter in current implementation. Reserved for future use.

    Returns:
        pd.DataFrame: The updated DataFrame with two new columns:
            - 'vectors': A dictionary of TF-IDF vectors (as lists) per feature.
            - 'vectorizers': A dictionary of fitted TfidfVectorizer objects per feature.

    Notes:
        - The function preprocesses text using lemmatization, stemming, and stop word removal.
        - It handles rows with invalid or empty JSON gracefully and logs errors.
        - Each feature's TF-IDF vector is generated using a separate `TfidfVectorizer` instance.
    """
    
    # Initialize new columns to store vectors and vectorizers
    df_tfidf['vectors'] = None
    df_tfidf['vectorizers'] = None
    lemmatizer, stemmer, stop_words = load_stemmer_lemmatizer_stopwords()

    for idx, row in df_tfidf.iterrows():
        vectors = {}       # To store actual TF-IDF vectors
        vectorizers = {}   # To store actual TfidfVectorizer objects
        column_value_as_string = str(row['Features'])

        # Check if the string is empty or just whitespace
        if column_value_as_string.strip():
            try:
                featuresValues = json.loads(column_value_as_string)

                # Iterate over features and generate TF-IDF vectors
                for categories in featuresValues:
                    for feature in categories.get("features", []):
                        name = feature.get('name', '')
                        description = feature.get('description', '')
                        description += ' ' + name + ' ' + categories.get('Category', '') + ' ' + row['main_category']
                        
                        if description.strip():
                            processed_description = preprocess_text(stemmer, lemmatizer, stop_words, description)
                            vectorizer = TfidfVectorizer()
                            tfidf_matrix = vectorizer.fit_transform([processed_description])
                            
                            # Store the vector and the vectorizer object
                            vectors[name] = tfidf_matrix.toarray().tolist()
                            vectorizers[name] = vectorizer
                        else:
                            vectors[name] = None
                            vectorizers[name] = None

                # Store in DataFrame
                df_tfidf.at[idx, 'vectors'] = vectors
                df_tfidf.at[idx, 'vectorizers'] = vectorizers

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for row {idx}: {e}")
                df_tfidf.at[idx, 'vectors'] = None
                df_tfidf.at[idx, 'vectorizers'] = None
        else:
            print(f"Empty or invalid JSON for row {idx}")
            df_tfidf.at[idx, 'vectors'] = None
            df_tfidf.at[idx, 'vectorizers'] = None

    return df_tfidf

def vectorize_data_withMongoSave(input_path, vectorizer_output_path, updated_vector_output_path):
    load_nltk_data()
    df = load_data(input_path)
    df_tfidf = df[['product_name', 'rating', 'seller', 'main_category', 'Features']]
    df_tfidf = generate_tfidf_per_row_withSave(df_tfidf.copy(), vectorizer_output_path)
    print(len(df_tfidf))
    df_tfidf.to_csv(updated_vector_output_path, index=False)
    vectorize_data_mongo(df_tfidf)
