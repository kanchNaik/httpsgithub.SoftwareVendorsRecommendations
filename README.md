Vendor Qualification System
1. Approach

The vendor qualification system identifies top vendors that align with a user’s software requirements. The approach is divided into several key steps:

Data Loading: The system loads a CSV file containing vendor information including product name, seller, rating, features, and main category.

Text Preprocessing: All text data is converted to lowercase, special characters are removed, and the text is tokenized. Stopwords are removed, and both lemmatization and stemming are applied to standardize the terms.

TF-IDF Vectorization: We use TF-IDF (Term Frequency-Inverse Document Frequency) to convert the cleaned text into numerical vectors. Importantly, vectorization is done for each feature separately, rather than combining all text into one long document. This avoids the dilution of similarity scores, especially when long descriptions would otherwise dominate shorter but more relevant features.

Similarity Scoring: The input query (based on software category and required capabilities) is also processed and vectorized. Cosine similarity is calculated between the query and each feature to assess relevance.

Filtering and Ranking: Only vendors with a similarity score above a chosen threshold are considered. These are then ranked based on their average similarity score across features. The top 10 ranked vendors are returned as the most qualified options.

2. Challenges Encountered

Data Cleanliness: Some rows had improperly formatted or missing JSON in the features column, requiring additional error handling.

Feature Length Bias: Initially, all feature text was vectorized as one block, which caused long features to dilute the similarity. This was solved by vectorizing each feature individually.

Cosine Similarity Issues: In a few cases, dimensional mismatches between query and feature vectors caused runtime issues. We resolved this with checks to ensure compatibility.

3. Potential Improvements

Use Precomputed Vectors: TF-IDF vectors could be stored and reused to reduce runtime during similarity scoring. I tried doing this but faced issues with reading the precomputed vectorizer again to vectorize the query with the same vectorizer as the feature. Respective code is present in the repository but due to time limitations could not spend more time on it.

Use More Advanced Embeddings: Instead of TF-IDF, using semantic vectorization models like BERT or Sentence-BERT could capture more nuanced relationships in language.

User Feedback Loop: Collecting user feedback on recommended vendors could help improve the model over time.

Dynamic Thresholding: Rather than a fixed cutoff, adaptive thresholds based on query specificity or vendor categories could improve relevance.
