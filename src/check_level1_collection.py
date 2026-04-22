from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
coll = client[os.getenv("MONGO_DB_NAME", "TopBrain_DB")]["HierarchicalPatients3D_Level1_CTA41"]

# Check 1 - structure des champs (sans les binaires)
doc = coll.find_one({}, {"img_data": 0, "lbl_data": 0, "mask_n0_data": 0})
print("=== Champs disponibles ===")
print(sorted(k for k in doc.keys() if k != "_id"))
print()

# Check 2 - date ingestion
first = coll.find_one({}, {"_id": 1})
print("=== Date ingestion (premier doc) ===")
print(first["_id"].generation_time)
print()

# Check 3 - mask_recall_vs_gt sur tous les docs
docs_meta = list(coll.find({}, {"_id": 0, "patient_id": 1, "mask_recall_vs_gt": 1, "mask_fg_ratio": 1}))
print("=== mask_recall_vs_gt par patient ===")
recalls = []
for d in docs_meta:
    rv = d.get("mask_recall_vs_gt")
    fr = d.get("mask_fg_ratio")
    recalls.append(rv)
    print(f"  {d.get('patient_id','?'):>6}: recall_vs_gt={rv}   fg_ratio={fr}")

valid = [r for r in recalls if r is not None]
if valid:
    import statistics
    print(f"\nmean={statistics.mean(valid):.4f}  min={min(valid):.4f}  max={max(valid):.4f}")

print(f"\nTotal docs: {coll.count_documents({})}")
client.close()
