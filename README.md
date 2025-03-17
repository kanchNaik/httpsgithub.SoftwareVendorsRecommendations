Vendor Qualification System
1. Approach

The vendor qualification system identifies top vendors that align with a user’s software requirements. The approach is divided into several key steps:

Data Loading: The system loads a CSV file containing vendor information including product name, seller, rating, features, and main category.

Text Preprocessing: All text data is converted to lowercase, special characters are removed, and the text is tokenized. Stopwords are removed, and both lemmatization and stemming are applied to standardize the terms.

TF-IDF Vectorization: We use TF-IDF (Term Frequency-Inverse Document Frequency) to convert the cleaned text into numerical vectors. Importantly, vectorization is done for each feature separately, rather than combining all text into one long document. This avoids the dilution of similarity scores, especially when long descriptions would otherwise dominate shorter but more relevant features.

Similarity Scoring: The input query (based on software category and required capabilities) is also processed and vectorized. Cosine similarity is calculated between the query and each feature to assess relevance.

Filtering and Ranking: Only vendors with a similarity score above a chosen threshold are considered. These are then ranked based on their average similarity score across features. The top 10 ranked vendors are returned as the most qualified options.

Code Flow -
- VendorQualification_With_TFIDF.ipynb -> 1st is a Python script in a collab file, where I was trying my TF-IDF approach - loading dataset, cleaning and pre-processing, computing TF-IDF, computing similarity based on cosine similarity, and finally ranking.
- The 2nd is the flask application with same code as above. I divided the code into multiple files and folders to make the architecture extensible for future use. code starts with the app.py file which contains the endpoint for API. VendorQualifier file has an orchestrator method that orchestrates all steps from reading data from the input file to ranking and filtering qualified vendors. CommonProcessingUtility contains methods that are generic and can be used in multiple places for leveraging common functionality. FullDataVectorizer contains a method that computes TF-IDF vectors and returns vectorized df. SimilarityEvaluator contains method which computes cosine similarity and filters df records based on similarity. These all files contain additional methods which I created to store pre-computed vectorizers to further enhance the model but I was encountering a few issues with re-loading vectorizers from pickle files and due to time constraints I could not give it more time. but it is further possibility to improve architecture.
- Ranking_vendors_by_BERT.ipynb ->  Similarity marking with deep learning model BERT - used distilBERT for computing similarity. Used attention mask while encoding to ignore irrelavent tokens. BERT is open source hence no cost for purchasing model like openAI.

2. Challenges Encountered

Data Cleanliness: Some rows had improperly formatted or missing JSON in the features column, requiring additional error handling.

Feature Length Bias: Initially, all feature text was vectorized as one block, which caused long features to dilute the similarity. This was solved by vectorizing each feature individually.

Cosine Similarity Issues: In a few cases, dimensional mismatches between query and feature vectors caused runtime issues. We resolved this with checks to ensure compatibility.

3. Potential Improvements

Use Precomputed Vectors: TF-IDF vectors could be stored and reused to reduce runtime during similarity scoring. I tried doing this but faced issues with reading the precomputed vectorizer again to vectorize the query with the same vectorizer as the feature. Respective code is present in the repository but due to time limitations could not spend more time on it.

Use More Advanced Embeddings: Instead of TF-IDF, using semantic vectorization models like BERT or Sentence-BERT could capture more nuanced relationships in language.

User Feedback Loop: Collecting user feedback on recommended vendors could help improve the model over time.

Dynamic Thresholding: Rather than a fixed cutoff, adaptive thresholds based on query specificity or vendor categories could improve relevance.

one of the approaches is using disltilBert model it is a lightweight model that performs similarly to BERT. we can train model and store it in pickle file and use it to predict similarity. since we are using a deep learning model, the model will itself learn features and we don't have to setup threshold manually. LoRA and QoRA fine-tunning techniques can be used to further optimize the model. I have added sample code, which is still not performing well but can be explored further to obtain better results. I also added some smoothing function but this model needs further work refer Ranking vendors by BERT.ipynb.
