import pickle
import os
import re

# Function to create a directory if it doesn't exist
create_directory = lambda path: os.makedirs(path, exist_ok=True)

# Function to clean the feature name as some features have special characters
def clean_feature_name(feature_name):
    feature_name = feature_name.lower()  # Convert to lowercase
    feature_name = re.sub(r'[^a-zA-Z0-9]', '', feature_name)  # Remove special characters and spaces
    return feature_name

# Function to save the vectorizer along with its vocabulary
def save_vectorizer(vectorizer, row_idx, feature_name, vectorizer_output_path):
    create_directory(vectorizer_output_path)
    cleaned_feature_name = clean_feature_name(feature_name)
    file_path = os.path.join(vectorizer_output_path, f"vectorizer_row{row_idx}_{cleaned_feature_name}.pkl")

    if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:
        print(f"Error: Vectorizer NOT fitted for feature '{feature_name}' in row {row_idx}. Skipping save.")
        return None  # Prevent saving an unfitted vectorizer

    # Save the entire vectorizer, not just the vocabulary
    with open(file_path, "wb") as file:
        pickle.dump(vectorizer, file)

    print(f"SAVING full vectorizer for feature '{feature_name}' in row {row_idx}")
    return file_path

# Function to load the vectorizer along with its vocabulary
def load_vectorizer(vectorizer_path):
    """Load the entire trained TfidfVectorizer object from file."""
    try:
        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)

        # Debug: Ensure the vectorizer is actually fitted
        if hasattr(vectorizer, 'idf_'):
            print(f"Successfully LOADED and RESTORED trained vectorizer from {vectorizer_path}")
        else:
            print(f"WARNING: Loaded vectorizer from {vectorizer_path} but it is NOT trained!")
            return None  # Avoid using an invalid vectorizer

        return vectorizer

    except Exception as e:
        print(f"ERROR loading vectorizer from {vectorizer_path}: {e}")
        return None  # Return None if loading failed

