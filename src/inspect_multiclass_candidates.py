from pymongo import MongoClient
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"), serverSelectionTimeoutMS=5000)
db = client[os.getenv("MONGO_DB_NAME", "TopBrain_DB")]
for name in ["patients", "PolygonPatients", "MultiClassPatients3D_Binary_CTA41", "Stage2_Cropped_4C"]:
    coll = db[name]
    doc = coll.find_one({}, {"_id": 0})
    if not doc:
        print(f"=== {name}: empty ===")
        continue
    print(f"=== {name} ===")
    print(sorted(doc.keys()))
    if "lbl_data" in doc and "shape" in doc:
        dtype = np.dtype(doc.get("lbl_dtype", "int64"))
        lbl = np.frombuffer(doc["lbl_data"], dtype=dtype).reshape(tuple(doc["shape"]))
        uniq = np.unique(lbl)
        print("lbl unique sample:", uniq[:20].tolist(), "max=", int(uniq.max()))
    print()
client.close()
