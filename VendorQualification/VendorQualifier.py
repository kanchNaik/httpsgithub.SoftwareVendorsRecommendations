from CommonProcessingUtility import load_nltk_data, load_data
from Vectorization.FullDataVectorizer import generate_tfidf_per_row
from SimilarityEvaluation.SimilarityEvaluator import calculate_similarity, filter_highly_similar_rows
from RankingService import rank_vendors
#from PreQualifiedList import set_prequalified_by_main_category

def get_qualifiedVendors(input_path, query, software_category, capabilities):
    """
    Filters and ranks vendors based on similarity to a query, within a specified software category,
    using TF-IDF vectorization and cosine similarity. The function returns the top 10 vendors based on
    their similarity scores and ranking.

    Args:
        input_path (str): Path to the input data file containing vendor information.
        query (str): The query text to compare against the vendors' features for similarity.
        software_category (str): The software category to filter the vendors by (case-insensitive).
        capabilities (dict): Additional capabilities or filters (not used in this implementation but reserved for future extensions).

    Returns:
        pd.DataFrame: A DataFrame containing the top 10 vendors sorted by their similarity to the query, including:
            - 'product_name': Name of the product/vendor.
            - 'rating': Vendor's rating.
            - 'seller': Vendor's seller.
            - 'main_category': Vendor's main software category.
            - 'Features': Features associated with the vendor.
            - 'avg_similarity_scores': The average similarity score of the vendor's features to the query.
            - 'final_score': Final score after ranking.
            - 'rank': Rank based on the final score.
    
    Notes:
        - The function preprocesses text and calculates TF-IDF vectors for the vendor features.
        - The vendors are filtered by their main category, then ranked based on their similarity to the input query.
        - The function assumes the input data is in a compatible format (e.g., CSV or JSON).
    """

    # Load vendor data from the specified file path
    df = load_data(input_path)

    # Select relevant columns from the data
    df = df[['product_name', 'rating', 'seller', 'main_category', 'Features']]

    # Filter vendors by the specified software category (case-insensitive)
    df = df[df['main_category'].str.contains(software_category, case=False, na=False)]

    # Load NLTK data required for text preprocessing
    load_nltk_data()

    # Generate TF-IDF vectors for each row based on the 'Features' column
    df = generate_tfidf_per_row(df.copy())

    # Calculate similarity scores between the query and vendor feature vectors
    df_new = calculate_similarity(query, df.copy())

    # Filter vendors that have highly similar features to the query (above a predefined threshold)
    filtereddf = filter_highly_similar_rows(df_new)

    # Rank the vendors based on their similarity scores
    rankedvendors = rank_vendors(filtereddf)

    # Return the top 10 vendors with the relevant information
    return rankedvendors[['product_name', 'rating', 'seller', 'main_category', 'Features', 'avg_similarity_scores', 'final_score', 'rank']].head(10)
