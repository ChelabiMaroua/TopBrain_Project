"""
app.py
======
Backend Flask TopBrain — API de segmentation vasculaire cérébrale.

Endpoints :
    POST /segment          Upload NIfTI -> lance inférence async -> retourne job_id
    GET  /status/<job_id>  Polling état du job (queued / running / done / error)
    GET  /result/<job_id>  Résultats JSON (metrics, class_volumes, clinical)
    GET  /download/<job_id>/<filename>  Télécharger un NIfTI de sortie

Installation :
    pip install flask flask-cors

Lancement :
    python backend/app.py
    python backend/app.py --ckpt-stage1 ... --ckpt-stage2 ... --ckpt-stage3 ...

Utilisation depuis l'interface :
    fetch('http://localhost:5000/segment', { method:'POST', body: formData })
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict

# Désactiver torch._dynamo avant tout import torch/monai
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

# Ajouter le répertoire parent au path pour importer inference_pipeline
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "5_HierarchicalSeg"))

from inference_pipeline import TopBrainPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration par défaut — peut être surchargé via CLI ou variables d'env
# ---------------------------------------------------------------------------
DEFAULT_CKPT_STAGE1 = str(ROOT / "4_Unet3D/checkpoints/stage1_binary_v2/swinunetr_best_fold_1.pth")
DEFAULT_CKPT_STAGE2 = str(ROOT / "5_HierarchicalSeg/checkpoints/stage3_level2_v2/swinunetr_level2_best_fold_1.pth")
DEFAULT_CKPT_STAGE3 = str(ROOT / "5_HierarchicalSeg/checkpoints/stage3_fine_v1/swinunetr_level2_best_fold_1.pth")
DEFAULT_UPLOAD_DIR  = str(ROOT / "results" / "flask_uploads")
DEFAULT_OUTPUT_DIR  = str(ROOT / "results" / "flask_outputs")
MAX_FILE_MB         = 512

# ---------------------------------------------------------------------------
# App Flask
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # autoriser les requêtes cross-origin depuis l'interface HTML

# State global — jobs en cours / terminés
jobs: Dict[str, Dict] = {}
jobs_lock = threading.Lock()

# Pipeline — chargé une seule fois au démarrage
pipeline: TopBrainPipeline | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return filename.lower().endswith((".nii", ".nii.gz"))


def get_job(job_id: str) -> Dict | None:
    with jobs_lock:
        return dict(jobs[job_id]) if job_id in jobs else None


def update_job(job_id: str, **kwargs) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)


def run_inference_async(job_id: str, nifti_path: str, output_dir: str) -> None:
    """Exécuté dans un thread séparé."""
    update_job(job_id, status="running", started_at=time.time())
    try:
        result = pipeline.run(nifti_input=nifti_path, output_dir=output_dir)
        update_job(
            job_id,
            status="done",
            finished_at=time.time(),
            metrics=result.metrics,
            class_volumes=result.class_volumes,
            clinical=result.clinical,
            timing=result.timing,
            original_shape=list(result.original_shape),
            outputs={
                "seg41":    result.seg_path.name,
                "families": result.family_path.name,
                "binary":   result.binary_path.name,
            },
            output_dir=output_dir,
        )
    except Exception as exc:
        update_job(
            job_id,
            status="error",
            finished_at=time.time(),
            error=str(exc),
        )
        import traceback
        print(f"[error] job {job_id}: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Sert l'interface HTML principale."""
    return send_from_directory(str(ROOT), "interface_medecin_v2.html")


@app.route("/health", methods=["GET"])
def health():
    """Vérification que le serveur est prêt."""
    return jsonify({
        "status":       "ok",
        "pipeline":     pipeline is not None,
        "jobs_active":  sum(1 for j in jobs.values() if j["status"] == "running"),
    })


@app.route("/segment", methods=["POST"])
def segment():
    """
    Upload un fichier NIfTI et lance la segmentation.

    Form-data :
        file : fichier .nii ou .nii.gz

    Retourne :
        { job_id, status, message }
    """
    if pipeline is None:
        return jsonify({"error": "Pipeline non initialisé"}), 503

    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier fourni (champ 'file' manquant)"}), 400

    f = request.files["file"]
    if not f.filename or not allowed_file(f.filename):
        return jsonify({"error": "Format invalide — fichier .nii ou .nii.gz requis"}), 400

    # Vérifier la taille
    f.seek(0, 2)
    size_mb = f.tell() / (1024 * 1024)
    f.seek(0)
    if size_mb > MAX_FILE_MB:
        return jsonify({"error": f"Fichier trop volumineux ({size_mb:.1f} MB > {MAX_FILE_MB} MB)"}), 413

    # Créer le job
    job_id     = str(uuid.uuid4())[:8]
    upload_dir = Path(DEFAULT_UPLOAD_DIR) / job_id
    output_dir = Path(DEFAULT_OUTPUT_DIR) / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le fichier uploadé
    safe_name  = Path(f.filename).name
    nifti_path = str(upload_dir / safe_name)
    f.save(nifti_path)

    # Enregistrer le job
    with jobs_lock:
        jobs[job_id] = {
            "job_id":      job_id,
            "status":      "queued",
            "filename":    safe_name,
            "created_at":  time.time(),
            "started_at":  None,
            "finished_at": None,
        }

    # Lancer l'inférence dans un thread
    t = threading.Thread(
        target=run_inference_async,
        args=(job_id, nifti_path, str(output_dir)),
        daemon=True,
    )
    t.start()

    return jsonify({
        "job_id":  job_id,
        "status":  "queued",
        "message": f"Segmentation lancée pour '{safe_name}'",
    }), 202


@app.route("/status/<job_id>", methods=["GET"])
def status(job_id: str):
    """
    Retourne l'état du job.

    Réponse :
        { job_id, status, elapsed_s }
        status ∈ { queued | running | done | error }
    """
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": f"Job '{job_id}' introuvable"}), 404

    elapsed = None
    if job.get("started_at"):
        end = job.get("finished_at") or time.time()
        elapsed = round(end - job["started_at"], 2)

    return jsonify({
        "job_id":    job_id,
        "status":    job["status"],
        "elapsed_s": elapsed,
        "error":     job.get("error"),
    })


@app.route("/result/<job_id>", methods=["GET"])
def result(job_id: str):
    """
    Retourne les résultats complets une fois le job terminé.

    Réponse :
        { job_id, metrics, class_volumes, clinical, timing, outputs, original_shape }
    """
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": f"Job '{job_id}' introuvable"}), 404

    if job["status"] != "done":
        return jsonify({
            "error":  f"Job non terminé (status={job['status']})",
            "status": job["status"],
        }), 409

    return jsonify({
        "job_id":         job_id,
        "status":         "done",
        "metrics":        job.get("metrics", {}),
        "class_volumes":  job.get("class_volumes", {}),
        "clinical":       job.get("clinical", {}),
        "timing":         job.get("timing", {}),
        "original_shape": job.get("original_shape"),
        "outputs":        job.get("outputs", {}),
        "elapsed_s":      round(job["finished_at"] - job["started_at"], 2) if job.get("finished_at") else None,
    })


@app.route("/download/<job_id>/<filename>", methods=["GET"])
def download(job_id: str, filename: str):
    """
    Télécharger un fichier NIfTI de sortie.

    Exemple : GET /download/abc12345/patient_seg41.nii.gz
    """
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Job introuvable"}), 404
    if job["status"] != "done":
        return jsonify({"error": "Job non terminé"}), 409

    # Sécurité — rejeter les path traversal
    safe = Path(filename).name
    file_path = Path(job["output_dir"]) / safe
    if not file_path.exists():
        return jsonify({"error": f"Fichier '{safe}' non trouvé"}), 404

    return send_file(str(file_path), as_attachment=True, download_name=safe)


@app.route("/jobs", methods=["GET"])
def list_jobs():
    """Liste tous les jobs (utile pour debug)."""
    with jobs_lock:
        summary = [
            {"job_id": j["job_id"], "status": j["status"], "filename": j.get("filename")}
            for j in jobs.values()
        ]
    return jsonify(summary)


# ---------------------------------------------------------------------------
# Entrée principale
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TopBrain Flask backend")
    parser.add_argument("--ckpt-stage1", default=DEFAULT_CKPT_STAGE1)
    parser.add_argument("--ckpt-stage2", default=DEFAULT_CKPT_STAGE2)
    parser.add_argument("--ckpt-stage3", default=DEFAULT_CKPT_STAGE3)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp",    action="store_true", default=True)
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=5000)
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    global pipeline
    print("[TopBrain] Initialisation du pipeline …")
    pipeline = TopBrainPipeline(
        ckpt_stage1=args.ckpt_stage1,
        ckpt_stage2=args.ckpt_stage2,
        ckpt_stage3=args.ckpt_stage3,
        device=args.device,
        amp=args.amp,
    )
    print(f"[TopBrain] Serveur prêt sur http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()

