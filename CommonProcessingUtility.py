import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

def load_nltk_data():
    """
    Downloads necessary NLTK datasets for text preprocessing, including:
    - 'punkt' for tokenization
    - 'punkt_tab' (which seems to be unnecessary here, can be removed)
    - 'stopwords' for removing common words from text
    - 'wordnet' for lemmatization

    This function is necessary to prepare the NLTK resources before text processing.
    """
    nltk.download('punkt')
    nltk.download('punkt_tab')  # Note: This download may not be required
    nltk.download('stopwords')
    nltk.download('wordnet')

def load_stemmer_lemmatizer_stopwords():
    """
    Loads and returns instances of the NLTK lemmatizer, stemmer, and stopwords set.

    Returns:
        tuple: Contains:
            - lemmatizer (WordNetLemmatizer): NLTK lemmatizer for word normalization.
            - stemmer (PorterStemmer): NLTK Porter stemmer for word stemming.
            - stop_words (set): Set of common English stopwords from NLTK.
    """
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    return lemmatizer, stemmer, stop_words

def load_data(file_path):
    """
    Loads a CSV file into a Pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file to be loaded.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_text(stemmer, lemmatizer, stop_words, text):
    """
    Preprocesses the input text by:
    - Converting to lowercase.
    - Removing non-alphanumeric characters.
    - Tokenizing the text into words.
    - Removing stopwords.
    - Lemmatizing and stemming the words.

    Args:
        stemmer (PorterStemmer): The stemmer to apply to each word.
        lemmatizer (WordNetLemmatizer): The lemmatizer to apply to each word.
        stop_words (set): Set of stopwords to be removed from the text.
        text (str): The input text to be processed.

    Returns:
        str: The processed text as a single string, where each word is lemmatized, stemmed, and non-stopword.
    """
    # Lowercase the text
    text = text.lower()

    # Remove special characters (non-alphanumeric characters)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenization (split the text into words)
    words = text.split()

    # Remove stopwords and apply stemming and lemmatization
    processed_words = []
    for word in words:
        if word not in stop_words:
            # Lemmatize and stem
            lemmatized_word = lemmatizer.lemmatize(word)
            stemmed_word = stemmer.stem(lemmatized_word)
            processed_words.append(stemmed_word)

    # Join the processed words back into a string
    return ' '.join(processed_words)

def clean_json_for_csv(json_data):
    """
    Cleans up JSON data by removing newline and carriage return characters to ensure proper CSV formatting.

    Args:
        json_data (str): The JSON data to be cleaned.

    Returns:
        str: The cleaned JSON data.
    """
    return json_data.replace('\n', ' ').replace('\r', ' ')

def get_query(software_category, capabilities):
    """
    Constructs a query string based on the provided software category and capabilities.

    If capabilities are provided, they are included in the query string. If only one capability is provided,
    it will be added directly; otherwise, the capabilities will be joined with commas and 'and' for the last one.
    
    Args:
        software_category (str): The category of software.
        capabilities (list): A list of capabilities to filter by.

    Returns:
        str: A query string describing the software category and its capabilities (if any).
    """
    if capabilities:
        if len(capabilities) == 1:
            cap_str = capabilities[0]
        else:
            cap_str = ', '.join(capabilities[:-1]) + ' and ' + capabilities[-1]
        query = f"{software_category} with {cap_str}"
    else:
        query = software_category
    
    return query
