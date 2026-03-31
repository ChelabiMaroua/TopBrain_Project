import argparse
import json
import random
import sys
from pathlib import Path
from pymongo import MongoClient

def get_unique_patients_from_db(mongo_uri, db_name, collection_name):
    """Récupère la liste de tous les IDs patients uniques (ex: topcow_ct_001)."""
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        collection = db[collection_name]
        
        # On récupère les IDs distincts directement depuis MongoDB
        all_ids = collection.distinct("patient_id")
        client.close()
        return sorted(all_ids)
    except Exception as e:
        print(f"[ERREUR MongoDB] Impossible de récupérer les patients : {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Générer une partition K-Fold dynamique depuis MongoDB")
    
    # Paramètres de connexion
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--db-name", default="TopBrain_DB")
    parser.add_argument("--collection", default="MultiClassPatients")
    
    # Paramètres de partitionnement
    parser.add_argument("--k", type=int, default=5, help="Nombre de Folds")
    # Note : On utilise %% pour échapper le symbole % dans l'aide d'argparse
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Ratio pour le Hold-out (ex: 0.2 = 20%%)")
    parser.add_argument("--seed", type=int, default=42, help="Seed pour la reproductibilité")
    
    # Sortie
    parser.add_argument("--output", default="3_DATA_Partitionement/partition_materialized.json")
    
    args = parser.parse_args()

    # Fixer la graine aléatoire pour que le tirage soit toujours le même
    random.seed(args.seed)

    # 1. Extraction des patients depuis ta base de données
    all_patients = get_unique_patients_from_db(args.mongo_uri, args.db_name, args.collection)
    total_count = len(all_patients)
    
    if total_count == 0:
        print("[ERREUR] La base de données est vide. Vérifie ton ETL.")
        return

    print(f"[*] {total_count} patients détectés dans {args.db_name}.{args.collection}")

    # 2. Création du Hold-out (Test Set final)
    shuffled = all_patients.copy()
    random.shuffle(shuffled)
    
    test_count = int(total_count * args.test_ratio)
    if test_count == 0 and total_count > 0: test_count = 1 # Sécurité : au moins 1 patient en test
    
    holdout_test_set = sorted(shuffled[:test_count])
    kfold_pool = sorted(shuffled[test_count:])
    
    print(f"[*] Hold-out (Test final) : {len(holdout_test_set)} patients")
    print(f"[*] K-Fold Pool (Train/Val) : {len(kfold_pool)} patients")

    # 3. Génération des K-Folds (Cross-Validation)
    # On mélange le pool pour la répartition dans les folds
    random.shuffle(kfold_pool)
    folds_data = {}
    val_size = len(kfold_pool) // args.k
    
    for i in range(args.k):
        start = i * val_size
        # Pour le dernier fold, on prend tout le reste
        end = (i + 1) * val_size if i < args.k - 1 else len(kfold_pool)
        
        val_ids = sorted(kfold_pool[start:end])
        train_ids = sorted([p for p in kfold_pool if p not in val_ids])
        
        folds_data[f"fold_{i}"] = {
            "train": train_ids,
            "val": val_ids,
            "train_count": len(train_ids),
            "val_count": len(val_ids)
        }

    # 4. Construction du JSON final
    payload = {
        "metadata": {
            "total_patients": total_count,
            "k_folds": args.k,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "db_source": args.db_name
        },
        "holdout_test_set": holdout_test_set,
        "folds": folds_data
    }

    # Sauvegarde
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("-" * 30)
    print(f"[SUCCESS] Partition validée et sauvegardée.")
    print(f"-> Fichier : {args.output}")
    for f_name, f_info in folds_data.items():
        print(f"   - {f_name}: Train={f_info['train_count']} | Val={f_info['val_count']}")

if __name__ == "__main__":
    main()