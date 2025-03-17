def rank_vendors(df, weight_similarity=0.7, weight_rating=0.3):
    """
    Ranks vendors based on weighted average of similarity and rating.
    
    Parameters:
    - df: DataFrame with columns `avg_similarity_scores`, `rating`, and optionally `prequalified`
    - weight_similarity: weight for average similarity score (default 0.7)
    - weight_rating: weight for vendor rating (default 0.3)
    - top_n: number of top vendors to return (default 10)
    
    Returns:
    - Ranked DataFrame with score and rank
    """
    df_ranked = df.copy()

    # Fill missing ratings with 0 if any
    if 'rating' in df_ranked.columns:
        df_ranked['rating'] = df_ranked['rating'].fillna(0)
    else:
        df_ranked['rating'] = 0  # Add rating column if not present

    # Normalize both columns to bring them into [0,1] range
    df_ranked['normalized_similarity'] = df_ranked['avg_similarity_scores'] / df_ranked['avg_similarity_scores'].max()
    df_ranked['normalized_rating'] = df_ranked['rating'] / df_ranked['rating'].max() if df_ranked['rating'].max() > 0 else 0

    # Compute final score using weighted sum
    df_ranked['final_score'] = (
        weight_similarity * df_ranked['normalized_similarity'] +
        weight_rating * df_ranked['normalized_rating']
    )


    # Sort: Prequalified vendors first, then by final score
    df_ranked.sort_values(by=['final_score'], ascending=[False], inplace=True)
    df_ranked['rank'] = range(1, len(df_ranked) + 1)

    return df_ranked
