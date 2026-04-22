from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"), serverSelectionTimeoutMS=5000)
db = client[os.getenv("MONGO_DB_NAME", "TopBrain_DB")]
for name in sorted(db.list_collection_names()):
    print(f"{name}: {db[name].count_documents({})}")
client.close()
