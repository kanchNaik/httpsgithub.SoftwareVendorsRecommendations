import pymongo

def get_mongo_client(mongo_uri):
    """Returns a MongoDB client connected to the given URI."""
    try:
        client = pymongo.MongoClient(mongo_uri)
        return client
    except pymongo.errors.ConnectionError as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def get_db(client, db_name):
    """Returns the database instance."""
    if client:
        return client[db_name]
    else:
        print("No client available, returning None")
        return None

def get_collection(db, collection_name):
    """Returns the collection instance."""
    if db:
        return db[collection_name]
    else:
        print(f"Database {db} not found, returning None")
        return None
