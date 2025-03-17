Vendor Qualification System
Approach

Data Loading:

The vendor data is loaded from a CSV file containing various features like product name, rating, seller, main category, and detailed feature descriptions.

Text Preprocessing:

  Text normalization: All text data (such as product descriptions and features) is preprocessed by converting it to lowercase, removing special characters, and tokenizing the text.
  Stopwords removal: Commonly used words that don’t provide much meaning (e.g., “and”, “the”, etc.) are removed from the text.
  Lemmatization and Stemming: Words are reduced to their root forms using both a lemmatizer and a stemmer to improve consistency across similar terms (e.g., “running” becomes “run”).

TF-IDF Vectorization:

A TF-IDF (Term Frequency-Inverse Document Frequency) approach is used to transform the feature descriptions into vectors. Vectorization is done for each feature separately, as treating all features as one long string would dilute the similarity scores, especially when features have vastly different lengths. This ensures that each feature's importance is captured independently, enabling more accurate similarity scoring between the query and each individual feature.

Similarity Scoring:

For each vendor, the cosine similarity between the processed query and the feature descriptions is computed. Cosine similarity measures how similar two vectors are, ranging from 0 (completely different) to 1 (identical).

Filtering and Ranking:

Vendors with similarity scores above a certain threshold (e.g., 0.6) are considered to be highly relevant.
The vendors are ranked based on their average similarity scores, with the most relevant vendors appearing at the top.
A final score is calculated for each vendor based on the similarity across multiple features.

Final Output:

The top 10 vendors with the highest scores are returned, providing the best matches to the given query.
