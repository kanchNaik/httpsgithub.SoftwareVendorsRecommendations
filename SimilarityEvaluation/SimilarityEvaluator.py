from CommonProcessingUtility import preprocess_text, load_stemmer_lemmatizer_stopwords
from sklearn.metrics.pairwise import cosine_similarity
from Vectorization.VectorizerUtility import load_vectorizer
import json
import numpy as np

def calculate_similarity_withMongo(query, df):
    lemmatizer, stemmer, stop_words = load_stemmer_lemmatizer_stopwords()
    processed_query = preprocess_text(stemmer, lemmatizer, stop_words, query)  # Preprocess the query

    similarity_results = []
    avg_similarity_scores = []

    for idx, row in df.iterrows():
        similarity_scores = {}

        # Load the feature vectors and vectorizer paths
        try:
            feature_vectors = row['vectors']  # Dictionary of feature vectors stored as JSON
            vectorizer_paths = row['vectorizer_paths']  # Dictionary of vectorizer paths stored as JSON
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error loading feature vectors or vectorizer paths for row {idx}: {e}")
            feature_vectors = None
            vectorizer_paths = None

        if feature_vectors and vectorizer_paths:
            total_score = 0
            count = 0

            for feature_name, feature_data in feature_vectors.items():
                print(f"Processing feature '{feature_name}' in row {idx}")

                if feature_data and 'vector' in feature_data:
                    try:
                        vectorizer_path = vectorizer_paths.get(feature_name, None)  # Get the path for the current feature

                        if vectorizer_path:
                            vectorizer = load_vectorizer(vectorizer_path)  # Load saved vectorizer

                            if not hasattr(vectorizer, 'vocabulary_'):
                                print(f"Error: Vectorizer not fitted for feature '{feature_name}' in row {idx}")
                                continue  # Skip this feature if vectorizer is not fitted

                            # Transform the query using the stored vectorizer
                            query_vector = vectorizer.transform([processed_query])
                            print(f"Query Vector Shape: {query_vector.shape}")

                            # Get stored feature vector and ensure it's in the correct shape
                            feature_vector = np.array(feature_data['vector']).reshape(1, -1)

                            # Check if shapes match before computing similarity
                            if query_vector.shape[1] == feature_vector.shape[1]:
                                similarity = cosine_similarity(query_vector, feature_vector)[0][0]
                            else:
                                print(f"Dimension mismatch in row {idx}, feature '{feature_name}'")
                                similarity = 0  # Default to 0 in case of dimension mismatch

                            total_score += similarity
                            count += 1
                            similarity_scores[feature_name] = similarity

                    except Exception as e:
                        print(f"Error processing feature '{feature_name}' in row {idx}: {e}")
                        similarity_scores[feature_name] = 0
                else:
                    similarity_scores[feature_name] = 0  # Handle missing vector or vectorizer path

            avg_similarity_scores.append(total_score / count if count > 0 else 0)
            similarity_results.append(similarity_scores)
        else:
            avg_similarity_scores.append(0)
            similarity_results.append({})  # Empty similarity score for this row

    df.drop(['vectors', 'vectorizer_paths'], axis=1, inplace=True)
    # Add the results to the DataFrame
    df['similarity_scores'] = similarity_results
    df['avg_similarity_scores'] = avg_similarity_scores

    return df

def calculate_similarity(query, df):
    """
    Calculates similarity scores between a preprocessed user query and text-based feature vectors 
    stored in a DataFrame for each row.

    Args:
        query (str): The input query string to compare against feature vectors.
        df (pd.DataFrame): A DataFrame containing the following columns:
            - 'vectors': A dictionary of precomputed feature vectors for each row.
            - 'vectorizers': A dictionary of fitted vectorizer objects corresponding to each feature.

    Returns:
        pd.DataFrame: The original DataFrame with the following added columns:
            - 'similarity_scores': A dictionary of cosine similarity scores per feature.
            - 'avg_similarity_scores': The average similarity score across all features in a row.

    Notes:
        - The query is first preprocessed using stemming, lemmatization, and stop word removal.
        - Cosine similarity is used to compare the vectorized query with stored feature vectors.
        - Handles missing vectorizers, unfitted vectorizers, and dimension mismatches.
        - Drops 'vectors' and 'vectorizers' columns before returning the final DataFrame.
    """

    lemmatizer, stemmer, stop_words = load_stemmer_lemmatizer_stopwords()
    processed_query = preprocess_text(stemmer, lemmatizer, stop_words, query)

    similarity_results = []
    avg_similarity_scores = []

    for idx, row in df.iterrows():
        similarity_scores = {}

        feature_vectors = row.get('vectors', {})
        feature_vectorizers = row.get('vectorizers', {})

        if feature_vectors and feature_vectorizers:
            total_score = 0
            count = 0

            for feature_name, vector in feature_vectors.items():
                print(f"Processing feature '{feature_name}' in row {idx}")

                if vector:
                    try:
                        vectorizer = feature_vectorizers.get(feature_name, None)

                        if vectorizer is None:
                            print(f"No vectorizer for feature '{feature_name}' in row {idx}")
                            similarity_scores[feature_name] = 0
                            continue

                        if not hasattr(vectorizer, 'vocabulary_'):
                            print(f"Error: Vectorizer not fitted for feature '{feature_name}' in row {idx}")
                            similarity_scores[feature_name] = 0
                            continue

                        # Transform the query
                        query_vector = vectorizer.transform([processed_query])
                        print(f"Query Vector Shape: {query_vector.shape}")

                        # Convert stored feature vector to numpy array
                        feature_vector = np.array(vector).reshape(1, -1)

                        if query_vector.shape[1] == feature_vector.shape[1]:
                            similarity = cosine_similarity(query_vector, feature_vector)[0][0]
                        else:
                            print(f"Dimension mismatch in row {idx}, feature '{feature_name}'")
                            similarity = 0

                        total_score += similarity
                        count += 1
                        similarity_scores[feature_name] = similarity

                    except Exception as e:
                        print(f"Error processing feature '{feature_name}' in row {idx}: {e}")
                        similarity_scores[feature_name] = 0
                else:
                    similarity_scores[feature_name] = 0

            #avg similarity scores calculations
            avg_similarity_scores.append(total_score / count if count > 0 else 0)
            similarity_results.append(similarity_scores)
        else:
            avg_similarity_scores.append(0)
            similarity_results.append({})

    #droping vectors once their work is done
    df.drop(['vectors', 'vectorizers'], axis=1, inplace=True)

    df['similarity_scores'] = similarity_results
    df['avg_similarity_scores'] = avg_similarity_scores

    return df

def filter_highly_similar_rows(df, threshold=0.6):
    """
    Filters rows in the DataFrame where at least one feature's similarity score 
    is greater than or equal to the given threshold.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'similarity_scores' column
            (a dictionary of similarity scores per feature) and 'avg_similarity_scores'.
        threshold (float, optional): The minimum similarity score to consider a row relevant.
            Defaults to 0.6.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the rows with at least one
        feature whose similarity score meets or exceeds the threshold, sorted by 
        'avg_similarity_scores' in descending order.
    """

    # Function to check if any feature in a row has similarity >= threshold
    def has_high_similarity(similarity_scores):
        return any(score >= threshold for score in similarity_scores.values())

    # Filter the DataFrame
    filtered_df = df[df['similarity_scores'].apply(has_high_similarity)]

    return filtered_df.sort_values(by='avg_similarity_scores', ascending=False)

    
    