from pymongo import MongoClient
import os
import numpy as np
import nibabel as nib
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"), serverSelectionTimeoutMS=5000)
db = client[os.getenv("MONGO_DB_NAME", "TopBrain_DB")]
doc = db["MultiClassPatients3D_Binary_CTA41"].find_one({}, {"_id": 0, "patient_id": 1, "lbl_path": 1})
print("patient_id:", doc.get("patient_id"))
print("lbl_path:", doc.get("lbl_path"))
path = doc.get("lbl_path")
if path and os.path.exists(path):
    arr = np.asarray(nib.load(path).get_fdata())
    uniq = np.unique(arr.astype(np.int64))
    print("exists: yes")
    print("unique sample:", uniq[:20].tolist())
    print("max:", int(uniq.max()))
else:
    print("exists: no")
client.close()
