from mongodbConnectio import get_mongo_client, get_db, get_collection
import pandas as pd
import json

def vectorize_data_mongo(df_tfidf):
    mongo_uri = "mongodb://localhost:27017/"
    collection_name = "vectorsforg2"
    db_name = "vectorsforg2db"

    # Get MongoDB connection
    client = get_mongo_client(mongo_uri)
    db = get_db(client, db_name)
    collection = get_collection(db, collection_name)
    
    if collection:
        # Convert DataFrame to dictionary format for MongoDB insertion
        data_dict = df_tfidf.to_dict(orient='records')  # Convert each row to a dictionary
        
        # Insert the data into MongoDB
        collection.insert_many(data_dict)  # Insert all records
        
        print(f"Inserted {len(data_dict)} records into MongoDB collection {collection_name}")

        # Close MongoDB connection
        client.close()
    else:
        print("Error: Could not establish MongoDB connection.")


def fetch_data_from_mongodb(query={}):
    # Connect to MongoDB
    mongo_uri = "mongodb://localhost:27017/"
    collection_name = "vectorsforg2"
    db_name = "vectorsforg2db"
    client = get_mongo_client(mongo_uri)
    db = get_db(client, db_name)
    collection = get_collection(db, collection_name)

    # Fetch documents
    cursor = collection.find(query)

    data = []
    for doc in cursor:
        doc['_id'] = str(doc['_id'])  # Convert ObjectId to string

        # Parse JSON string fields if necessary
        for field in ['Features', 'vectors', 'vectorizer_paths']:
            if field in doc and isinstance(doc[field], str):
                try:
                    doc[field] = json.loads(doc[field])
                except Exception as e:
                    print(f"Failed to parse {field}: {e}")

        data.append(doc)

    return pd.DataFrame(data)
