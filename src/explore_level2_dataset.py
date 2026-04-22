"""
explore_level2_dataset.py
=========================
Analyse statistique du dataset Level-2 (stage-3 : 41 classes) à partir des
fichiers NIfTI de labels bruts (lbl_path dans la collection MongoDB source).

Génère :
  1. Distribution des volumes GT (voxels) par classe par patient
  2. Taux de présence par classe (% de patients où la classe est présente)
  3. Classement des classes par volume moyen
  4. Comparaison avec la hiérarchie Level-1 (5 familles)
  5. Graphiques de distribution (sauvegardés dans results/plots/)

Usage :
    python explore_level2_dataset.py
    python explore_level2_dataset.py --max-patients 10    # test rapide
    python explore_level2_dataset.py --output-json results/level2_stats.json
    python explore_level2_dataset.py --plots              # génère les graphiques
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "1_ETL" / "Transform"))
sys.path.insert(0, str(ROOT / "ETL" / "Transform"))

# ─── Mapping Level-1 (familles) ───────────────────────────────────────────────
# 0 = BG, 1 = CoW (1-10), 2 = Ant/Mid (11-20), 3 = Post (21-34), 4 = Vein (35-40)
FAMILY_LUT: Dict[int, int] = {0: 0}
for i in range(1,  11): FAMILY_LUT[i] = 1   # CoW
for i in range(11, 21): FAMILY_LUT[i] = 2   # Ant/Mid
for i in range(21, 35): FAMILY_LUT[i] = 3   # Post
for i in range(35, 41): FAMILY_LUT[i] = 4   # Vein

FAMILY_NAMES = {0: "BG", 1: "CoW (1-10)", 2: "Ant/Mid (11-20)",
                3: "Post (21-34)", 4: "Vein (35-40)"}

# Noms anatomiques des 40 classes (basés sur TopCow / nomenclature ICA)
# Source : TopCow challenge paper + nomenclature neuro-anatomie standard
CLASS_LABELS: Dict[int, str] = {
    0:  "Background",
    # ── Famille CoW (1-10) ─────────────────────────────────────────────────
    1:  "ICA-L (A-ICA gauche)",
    2:  "ICA-R (A-ICA droite)",
    3:  "MCA-L M1 (Sylvienne gauche M1)",
    4:  "MCA-R M1 (Sylvienne droite M1)",
    5:  "ACA-L A1 (Cérébrale ant. gauche A1)",
    6:  "ACA-R A1 (Cérébrale ant. droite A1)",
    7:  "ACom (Communicante antérieure)",
    8:  "PCom-L (Communicante post. gauche)",
    9:  "PCom-R (Communicante post. droite)",
    10: "BA (Tronc basilaire)",
    # ── Famille Ant/Mid (11-20) ────────────────────────────────────────────
    11: "MCA-L M2+ (Sylvienne gauche M2+)",
    12: "MCA-R M2+ (Sylvienne droite M2+)",
    13: "ACA-L A2 (Cérébrale ant. gauche A2)",
    14: "ACA-R A2 (Cérébrale ant. droite A2)",
    15: "ACA-L A3+ (Cérébrale ant. gauche A3+)",
    16: "ACA-R A3+ (Cérébrale ant. droite A3+)",
    17: "Heubner-L (Récurrente de Heubner gauche)",
    18: "Heubner-R (Récurrente de Heubner droite)",
    19: "Perfo-ant-L (Perforantes ant. gauches)",
    20: "Perfo-ant-R (Perforantes ant. droites)",
    # ── Famille Post (21-34) ───────────────────────────────────────────────
    21: "PCA-L P1 (Cérébrale post. gauche P1)",
    22: "PCA-R P1 (Cérébrale post. droite P1)",
    23: "PCA-L P2+ (Cérébrale post. gauche P2+)",
    24: "PCA-R P2+ (Cérébrale post. droite P2+)",
    25: "SCA-L (Cérébelleuse sup. gauche)",
    26: "SCA-R (Cérébelleuse sup. droite)",
    27: "AICA-L (Cérébelleuse ant-inf. gauche)",
    28: "AICA-R (Cérébelleuse ant-inf. droite)",
    29: "PICA-L (Cérébelleuse post-inf. gauche)",
    30: "PICA-R (Cérébelleuse post-inf. droite)",
    31: "VA-L (Vertébrale gauche)",
    32: "VA-R (Vertébrale droite)",
    33: "Perfo-post-L (Perforantes post. gauches)",
    34: "Perfo-post-R (Perforantes post. droites)",
    # ── Famille Vein (35-40) ───────────────────────────────────────────────
    35: "SSS (Sinus sagittal sup.)",
    36: "TorHer (Torcular Herophili)",
    37: "TS-L (Sinus transverse gauche)",
    38: "TS-R (Sinus transverse droit)",
    39: "SigS-L (Sinus sigmoïde gauche)",
    40: "SigS-R (Sinus sigmoïde droit)",
}

HEADER_LINE = "─" * 90


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse statistique dataset Level-2 (41 classes)")
    p.add_argument("--collection", default="MultiClassPatients3D_Binary_CTA41",
                   help="Collection MongoDB source (contient lbl_path)")
    p.add_argument("--target-size", default="128x128x64")
    p.add_argument("--max-patients", type=int, default=0,
                   help="0 = tous les patients. N>0 = limiter pour test rapide.")
    p.add_argument("--output-json", default="results/level2_stats.json")
    p.add_argument("--plots", action="store_true",
                   help="Générer les graphiques (requiert matplotlib)")
    p.add_argument("--presence-threshold", type=float, default=0.80,
                   help="Seuil de présence pour critère C (défaut 0.80 = 80%% des patients)")
    return p.parse_args()


# ─── MongoDB ──────────────────────────────────────────────────────────────────
def fetch_lbl_paths(collection_name: str, target_size: str, max_patients: int) -> List[Dict]:
    from dotenv import load_dotenv
    from pymongo import MongoClient

    load_dotenv()
    uri     = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB_NAME", "TopBrain_DB")
    client  = MongoClient(uri, serverSelectionTimeoutMS=5000)
    coll    = client[db_name][collection_name]

    query   = {"target_size": target_size}
    proj    = {"_id": 0, "patient_id": 1, "lbl_path": 1}
    cursor  = coll.find(query, proj)
    if max_patients > 0:
        cursor = cursor.limit(max_patients)
    docs = list(cursor)
    client.close()

    # Normalisation patient_id
    def norm(v: object) -> str:
        nums = re.findall(r"\d+", str(v))
        return nums[-1].zfill(3) if nums else str(v)

    for d in docs:
        d["patient_id"] = norm(d.get("patient_id", ""))
    return docs


# ─── Lecture labels NIfTI ─────────────────────────────────────────────────────
def load_nifti_label(path: str) -> Optional[np.ndarray]:
    try:
        import nibabel as nib  # type: ignore
        img = nib.load(path)
        arr = np.asarray(img.dataobj, dtype=np.int32)
        return arr
    except Exception as e:
        print(f"  [warn] Impossible de charger {path} : {e}")
        return None


# ─── Statistiques ─────────────────────────────────────────────────────────────
def compute_patient_class_volumes(lbl: np.ndarray, n_classes: int = 41) -> np.ndarray:
    """Retourne un vecteur de n_classes entiers = nombre de voxels par classe."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        counts[c] = int(np.sum(lbl == c))
    return counts


def aggregate_stats(
    all_volumes: List[Tuple[str, np.ndarray]],   # (patient_id, volumes[41])
    n_classes: int = 41,
) -> Dict:
    """
    Calcule pour chaque classe :
      - mean_voxels  : volume moyen (patients où la classe est présente)
      - std_voxels   : écart-type du volume
      - n_present    : nombre de patients où la classe est présente (> 0 voxels)
      - pct_present  : % de présence
      - median_voxels
    """
    n_patients = len(all_volumes)
    stats: Dict[int, Dict] = {}

    all_arr = np.stack([v for _, v in all_volumes], axis=0)  # [N, 41]

    for c in range(n_classes):
        col    = all_arr[:, c]
        present = col > 0
        n_pres  = int(np.sum(present))
        vols_pres = col[present].astype(float)
        stats[c] = {
            "class_id":      c,
            "label":         CLASS_LABELS.get(c, f"class_{c}"),
            "family":        FAMILY_LUT.get(c, 0),
            "family_name":   FAMILY_NAMES.get(FAMILY_LUT.get(c, 0), "?"),
            "n_present":     n_pres,
            "pct_present":   round(100.0 * n_pres / n_patients, 1) if n_patients > 0 else 0.0,
            "mean_voxels":   round(float(np.mean(vols_pres)), 1) if n_pres > 0 else 0.0,
            "std_voxels":    round(float(np.std(vols_pres)),  1) if n_pres > 0 else 0.0,
            "median_voxels": round(float(np.median(vols_pres)), 1) if n_pres > 0 else 0.0,
            "min_voxels":    int(vols_pres.min()) if n_pres > 0 else 0,
            "max_voxels":    int(vols_pres.max()) if n_pres > 0 else 0,
        }
    return stats


# ─── Impression ───────────────────────────────────────────────────────────────
def print_full_table(stats: Dict, n_patients: int, presence_thresh: float) -> None:
    thresh_pct = 100.0 * presence_thresh
    print(f"\n{HEADER_LINE}")
    print(f"DISTRIBUTION DES VOLUMES PAR CLASSE ({n_patients} patients)")
    print(f"Critère C de présence : ≥ {thresh_pct:.0f}% des patients")
    print(HEADER_LINE)
    print(f"  {'ID':>3}  {'Famille':<12} {'Label (abrégé)':<32} "
          f"{'Présence':>9} {'Moy vox':>9} {'Médiane':>9} {'σ':>8}  Critère-C")
    print("  " + "-" * 88)

    for c in range(1, 41):
        s = stats[c]
        crit = "✓" if s["pct_present"] >= thresh_pct else " "
        label_short = s["label"][:32]
        print(f"  {c:>3}  {s['family_name']:<12} {label_short:<32} "
              f"{s['pct_present']:>7.1f}%  {s['mean_voxels']:>9.0f}  "
              f"{s['median_voxels']:>9.0f}  {s['std_voxels']:>8.0f}  {crit}")


def print_family_summary(stats: Dict, n_patients: int, presence_thresh: float) -> None:
    thresh_pct = 100.0 * presence_thresh
    print(f"\n{HEADER_LINE}")
    print("RÉSUMÉ PAR FAMILLE (Level-1)")
    print(HEADER_LINE)
    for fam_id, fam_name in FAMILY_NAMES.items():
        if fam_id == 0:
            continue
        members = [c for c, s in stats.items() if s["family"] == fam_id and c > 0]
        n_crit  = sum(1 for c in members if stats[c]["pct_present"] >= thresh_pct)
        total_vox = sum(stats[c]["mean_voxels"] for c in members)
        print(f"  {fam_name:<20}  {len(members):>2} sous-classes  "
              f"{n_crit:>2} répondent critère-C  "
              f"volume total moyen = {total_vox:>8.0f} vox")


def print_criterion_c_classes(stats: Dict, presence_thresh: float) -> List[int]:
    thresh_pct = 100.0 * presence_thresh
    crit_c = [c for c in range(1, 41) if stats[c]["pct_present"] >= thresh_pct]
    print(f"\n{HEADER_LINE}")
    print(f"CLASSES CRITÈRE-C (présentes chez ≥{thresh_pct:.0f}% des patients) — {len(crit_c)} classes")
    print(HEADER_LINE)
    for c in sorted(crit_c, key=lambda x: -stats[x]["mean_voxels"]):
        s = stats[c]
        print(f"  [{c:>2}]  {s['label']:<45}  "
              f"{s['pct_present']:>5.1f}%  {s['mean_voxels']:>8.0f} vox")
    return crit_c


def print_top_n_by_volume(stats: Dict, n: int = 15) -> List[int]:
    ranked = sorted(range(1, 41), key=lambda c: -stats[c]["mean_voxels"])[:n]
    print(f"\n{HEADER_LINE}")
    print(f"TOP-{n} CLASSES PAR VOLUME MOYEN (candidats pour Critère A)")
    print(HEADER_LINE)
    for rank, c in enumerate(ranked, 1):
        s = stats[c]
        print(f"  {rank:>2}. [{c:>2}]  {s['label']:<45}  "
              f"{s['mean_voxels']:>8.0f} vox  {s['pct_present']:>5.1f}%")
    return ranked


def print_small_classes(stats: Dict, vox_threshold: int = 200) -> List[int]:
    small = [c for c in range(1, 41) if 0 < stats[c]["mean_voxels"] < vox_threshold]
    small_sorted = sorted(small, key=lambda c: stats[c]["mean_voxels"])
    print(f"\n{HEADER_LINE}")
    print(f"CLASSES SUB-VOXEL / PETITES (volume moyen < {vox_threshold} vox) — {len(small)} classes")
    print(HEADER_LINE)
    print("  Ces classes sont candidates aux limitations de résolution.")
    for c in small_sorted:
        s = stats[c]
        print(f"  [{c:>2}]  {s['label']:<45}  {s['mean_voxels']:>6.0f} vox  {s['pct_present']:>5.1f}%")
    return small_sorted


# ─── Graphiques ───────────────────────────────────────────────────────────────
def make_plots(stats: Dict, n_patients: int, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib non disponible, --plots ignoré")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    classes = list(range(1, 41))
    means   = [stats[c]["mean_voxels"]   for c in classes]
    prcts   = [stats[c]["pct_present"]   for c in classes]
    colors  = [["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"][stats[c]["family"]]
               for c in classes]

    # ── Plot 1 : Volume moyen par classe ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 6))
    bars = ax.bar(classes, means, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Classe ID")
    ax.set_ylabel("Volume moyen (voxels)")
    ax.set_title(f"Volume GT moyen par classe — {n_patients} patients")
    ax.set_xticks(classes)
    ax.set_xticklabels([str(c) for c in classes], fontsize=7)
    ax.axhline(300, color="red", linestyle="--", linewidth=1, label="seuil 300 vox")
    ax.legend()
    fig.tight_layout()
    path1 = out_dir / "level2_volume_per_class.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path1.relative_to(ROOT)}")

    # ── Plot 2 : Taux de présence par classe ───────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(classes, prcts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Classe ID")
    ax.set_ylabel("% patients avec la classe")
    ax.set_title(f"Présence par classe (critère C) — {n_patients} patients")
    ax.set_xticks(classes)
    ax.set_xticklabels([str(c) for c in classes], fontsize=7)
    ax.axhline(80, color="red", linestyle="--", linewidth=1, label="seuil 80%")
    ax.legend()
    fig.tight_layout()
    path2 = out_dir / "level2_presence_per_class.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path2.relative_to(ROOT)}")

    # ── Plot 3 : Scatter volume vs présence ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(prcts, means, c=[stats[c]["family"] for c in classes],
                    cmap="tab10", s=60, alpha=0.85, edgecolors="white", linewidths=0.5)
    for c in classes:
        ax.annotate(str(c), (stats[c]["pct_present"], stats[c]["mean_voxels"]),
                    fontsize=6, ha="center", va="bottom")
    ax.axvline(80, color="red", linestyle="--", linewidth=1, label="critère-C 80%")
    ax.axhline(300, color="orange", linestyle="--", linewidth=1, label="critère-A 300 vox")
    ax.set_xlabel("% présence dans le dataset")
    ax.set_ylabel("Volume moyen (voxels)")
    ax.set_title("Volume vs Présence — chaque point = une classe")
    ax.legend()
    fig.tight_layout()
    path3 = out_dir / "level2_volume_vs_presence.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path3.relative_to(ROOT)}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    print(f"\n{HEADER_LINE}")
    print("EXPLORATION DATASET LEVEL-2 (41 classes)")
    print(HEADER_LINE)
    print(f"  Collection   : {args.collection}")
    print(f"  Target size  : {args.target_size}")
    print(f"  Max patients : {args.max_patients if args.max_patients > 0 else 'tous'}")

    # Fetch MongoDB
    print("\n[1] Récupération des lbl_path depuis MongoDB...")
    docs = fetch_lbl_paths(args.collection, args.target_size, args.max_patients)
    print(f"    {len(docs)} patients trouvés")
    if not docs:
        print("[erreur] Aucun document. Vérife la collection et le target_size.")
        return

    # Lecture labels NIfTI
    print("\n[2] Lecture des labels NIfTI et calcul des volumes par classe...")
    all_volumes: List[Tuple[str, np.ndarray]] = []
    failed: List[str] = []

    for doc in docs:
        pid      = doc["patient_id"]
        lbl_path = doc.get("lbl_path", "")
        if not lbl_path:
            print(f"  [warn] patient {pid} : pas de lbl_path")
            failed.append(pid)
            continue

        lbl = load_nifti_label(lbl_path)
        if lbl is None:
            failed.append(pid)
            continue

        max_lbl = int(lbl.max())
        if max_lbl < 2:
            print(f"  [warn] patient {pid} : max_label={max_lbl} → labels binaires, on ignore")
            failed.append(pid)
            continue

        vols = compute_patient_class_volumes(lbl, n_classes=41)
        all_volumes.append((pid, vols))
        print(f"  {pid}  max_lbl={max_lbl:>2}  total_fg_vox={vols[1:].sum():>8,}")

    n_valid = len(all_volumes)
    print(f"\n  {n_valid}/{len(docs)} patients valides  |  {len(failed)} échoués/ignorés")
    if failed:
        print(f"  Patients ignorés : {failed}")

    if n_valid == 0:
        print("[erreur] Aucun patient valide. Vérife lbl_path dans la collection.")
        return

    # Statistiques agrégées
    print("\n[3] Calcul des statistiques agrégées...")
    stats = aggregate_stats(all_volumes, n_classes=41)

    # Affichage
    print_full_table(stats, n_valid, args.presence_threshold)
    print_family_summary(stats, n_valid, args.presence_threshold)
    crit_c  = print_criterion_c_classes(stats, args.presence_threshold)
    top15   = print_top_n_by_volume(stats, n=15)
    small   = print_small_classes(stats, vox_threshold=200)

    # Résumé actionnable
    print(f"\n{HEADER_LINE}")
    print("RÉSUMÉ ACTIONNABLE POUR LA THÈSE")
    print(HEADER_LINE)
    print(f"  Dataset : {n_valid} patients  |  41 classes (0-40)  |  résolution 128×128×64")
    print(f"  Classes avec présence ≥ {100*args.presence_threshold:.0f}% (Critère C) : {len(crit_c)} classes")
    print(f"  Classes avec volume moyen < 200 vox (sub-voxel) : {len(small)} classes")
    print(f"  Intersection Critère-C ∩ Top-15 volume : "
          f"{len(set(crit_c) & set(top15))} classes")
    print()
    print("  Prochaine étape : définir class_hierarchy.json (Critère B, pertinence clinique)")
    print("  puis lancer l'ingestion Level-2 après CV5 stage-2.")

    # Plots
    if args.plots:
        print("\n[4] Génération des graphiques...")
        make_plots(stats, n_valid, ROOT / "results" / "plots")

    # Sauvegarde JSON
    if args.output_json:
        # Convertir stats en liste sérialisable
        stats_list = [stats[c] for c in range(41)]
        report = {
            "n_patients":       n_valid,
            "collection":       args.collection,
            "target_size":      args.target_size,
            "presence_threshold": args.presence_threshold,
            "class_stats":      stats_list,
            "criterion_c_ids":  crit_c,
            "top15_by_volume":  top15,
            "small_classes_ids": [int(c) for c in small],
            "per_patient_volumes": [
                {"patient_id": pid, "volumes": vols.tolist()}
                for pid, vols in all_volumes
            ],
        }
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[done] Statistiques Level-2 sauvegardées → {out}")
        print(f"       Utilise ce fichier pour définir class_hierarchy.json")


if __name__ == "__main__":
    main()
