"""Quick diagnostic for patient 015 Vein anomaly."""
from pymongo import MongoClient
import numpy as np, os, re
from dotenv import load_dotenv
load_dotenv()

VAL_PIDS = ["008", "015", "023", "027"]
CLASS_NAMES = {1: "CoW", 2: "Ant/Mid", 3: "Post", 4: "Vein"}

def norm(v):
    nums = re.findall(r"\d+", str(v))
    return nums[-1].zfill(3) if nums else str(v)

coll = MongoClient(os.getenv("MONGO_URI"))[os.getenv("MONGO_DB_NAME")]["HierarchicalPatients3D_Level1_CTA41"]
docs = list(coll.find({}, {"patient_id": 1, "shape": 1,
                            "lbl_data": 1, "lbl_dtype": 1,
                            "mask_n0_data": 1, "mask_n0_dtype": 1}))
print(f"Total docs fetched: {len(docs)}\n")

# ── Hypothèse 1 : GT voxel counts ────────────────────────────────────────────
print("=== Hypothese 1 : GT voxel counts par patient val ===")
print(f"  {'pid':<6} {'BG':>8} {'CoW':>8} {'Ant/Mid':>9} {'Post':>8} {'Vein':>8}  "
      f"{'Vein%fg':>8}")
for pid in VAL_PIDS:
    doc = next((d for d in docs if norm(d.get("patient_id", "")) == pid), None)
    if doc is None:
        print(f"  {pid}: NOT FOUND"); continue
    shape = tuple(doc["shape"])
    lbl = np.frombuffer(doc["lbl_data"],
                        dtype=np.dtype(doc.get("lbl_dtype", "uint8"))).reshape(shape)
    counts = [int((lbl == c).sum()) for c in range(5)]
    fg = sum(counts[1:])
    vein_pct = 100 * counts[4] / max(fg, 1)
    print(f"  {pid:<6} {counts[0]:>8} {counts[1]:>8} {counts[2]:>9} "
          f"{counts[3]:>8} {counts[4]:>8}  {vein_pct:>7.1f}%")

# ── Hypothèse 2 : prior stage-1 recall par classe ────────────────────────────
print()
print("=== Hypothese 2 : prior stage-1 recall par classe ===")
print(f"  {'pid':<6} {'CoW':>8} {'Ant/Mid':>9} {'Post':>8} {'Vein':>8}")
for pid in VAL_PIDS:
    doc = next((d for d in docs if norm(d.get("patient_id", "")) == pid), None)
    if doc is None:
        print(f"  {pid}: NOT FOUND"); continue
    shape = tuple(doc["shape"])
    lbl  = np.frombuffer(doc["lbl_data"],
                         dtype=np.dtype(doc.get("lbl_dtype", "uint8"))).reshape(shape)
    mask = np.frombuffer(doc["mask_n0_data"],
                         dtype=np.dtype(doc.get("mask_n0_dtype", "uint8"))).reshape(shape).astype(bool)
    row = []
    for c in range(1, 5):
        gt_c = lbl == c
        if gt_c.sum() == 0:
            row.append("  ABSENT")
        else:
            rec = float((mask & gt_c).sum()) / float(gt_c.sum())
            row.append(f"{rec:>8.4f}")
    print(f"  {pid:<6} {'  '.join(row)}")

# ── Extra : pour 015, combien de Vein GT est inside vs outside du prior mask ─
print()
print("=== Focus patient 015 : Vein GT inside/outside prior mask ===")
doc015 = next((d for d in docs if norm(d.get("patient_id", "")) == "015"), None)
if doc015:
    shape = tuple(doc015["shape"])
    lbl  = np.frombuffer(doc015["lbl_data"],
                         dtype=np.dtype(doc015.get("lbl_dtype", "uint8"))).reshape(shape)
    mask = np.frombuffer(doc015["mask_n0_data"],
                         dtype=np.dtype(doc015.get("mask_n0_dtype", "uint8"))).reshape(shape).astype(bool)
    gt_vein = lbl == 4
    inside  = int((gt_vein & mask).sum())
    outside = int((gt_vein & ~mask).sum())
    total   = int(gt_vein.sum())
    print(f"  GT Vein total   = {total}")
    print(f"  Inside  prior   = {inside}  ({100*inside/max(total,1):.1f}%)")
    print(f"  Outside prior   = {outside}  ({100*outside/max(total,1):.1f}%)")
    print(f"  => If stage-2 only predicts inside prior, max achievable recall = {inside/max(total,1):.4f}")
else:
    print("  Patient 015 not found")
