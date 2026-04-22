"""
aggregate_cv5_results.py
========================
Agrège les résultats des 5 folds de la cross-validation stage-2 (Level-1)
et imprime un tableau récapitulatif par classe + métrique globale CV5.

Sources attendues :
  results/level1_diag_fold_1.json        ← fold_1 val (008, 015, 023, 027)
  results/level1_diag_fold_2_val.json    ← fold_2 val
  results/level1_diag_fold_3_val.json    ← fold_3 val
  results/level1_diag_fold_4_val.json    ← fold_4 val
  results/level1_diag_fold_5_val.json    ← fold_5 val
  results/level1_diag_test_final.json    ← holdout test (011,014,018,021,022)

Usage :
    python aggregate_cv5_results.py
    python aggregate_cv5_results.py --output-json results/cv5_summary.json
    python aggregate_cv5_results.py --latex          # tableau LaTeX pour la thèse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ─── Constantes ───────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "BG", 1: "CoW", 2: "Ant/Mid", 3: "Post", 4: "Vein"}
FG_CLASSES  = [1, 2, 3, 4]

ROOT = Path(__file__).resolve().parent

FOLD_JSON_MAP: Dict[str, Path] = {
    "fold_1": ROOT / "results" / "level1_diag_fold_1.json",
    "fold_2": ROOT / "results" / "level1_diag_fold_2_val.json",
    "fold_3": ROOT / "results" / "level1_diag_fold_3_val.json",
    "fold_4": ROOT / "results" / "level1_diag_fold_4_val.json",
    "fold_5": ROOT / "results" / "level1_diag_fold_5_val.json",
}
TEST_JSON = ROOT / "results" / "level1_diag_test_final.json"

HEADER_LINE = "─" * 80


# ─── Chargement ───────────────────────────────────────────────────────────────
def load_fold_report(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_per_patient_metrics(report: Dict) -> List[Dict]:
    """Extrait la liste des métriques par patient depuis un rapport JSON."""
    return report.get("per_patient", [])


def aggregate_patients(patients: List[Dict]) -> Dict[str, List[float]]:
    """Accumule les métriques numériques de tous les patients."""
    agg: Dict[str, List[float]] = {}
    for p in patients:
        for k, v in p.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                agg.setdefault(k, []).append(float(v))
    return agg


# ─── Formatage ────────────────────────────────────────────────────────────────
def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _verdict(dm: float, ds: float, rm: float) -> str:
    if np.isnan(dm):
        return "⚠  absent"
    if dm < 0.40:
        return "🔴 <0.40"
    if dm < 0.55:
        return "🟡 <0.55"
    if ds > 0.15:
        return "🟠 σ élevé"
    if rm < 0.50:
        return "🟡 recall bas"
    return "🟢 OK"


# ─── Impression tableaux ──────────────────────────────────────────────────────
def print_fold_summary(fold: str, patients: List[Dict]) -> None:
    agg = aggregate_patients(patients)
    n   = len(patients)
    print(f"\n  {fold}  ({n} patients : "
          f"{', '.join(str(p.get('patient_id', '?')) for p in patients)})")
    for c in FG_CLASSES:
        dv = agg.get(f"dice_class_{c}", [])
        dm = float(np.mean(dv)) if dv else float("nan")
        ds = float(np.std(dv))  if len(dv) > 1 else 0.0
        print(f"    {CLASS_NAMES[c]:<10}  Dice={_fmt(dm)}  σ={_fmt(ds)}")
    fg = agg.get("mean_dice_fg", [])
    print(f"    {'FG mean':<10}  Dice={_fmt(float(np.mean(fg)) if fg else float('nan'))}")


def print_cv5_table(
    cv5_agg: Dict[str, Dict[str, List[float]]],
    test_agg: Optional[Dict[str, List[float]]],
) -> None:
    """Tableau agrégé global : une ligne par classe, une colonne par fold + CV5 + test."""
    folds_available = sorted(cv5_agg.keys())

    print(f"\n{HEADER_LINE}")
    print("TABLEAU CV5 — Dice par classe et par fold")
    print(HEADER_LINE)

    # En-tête
    header = f"  {'Classe':<10}"
    for f in folds_available:
        header += f"  {f:>8}"
    header += f"  {'CV5 μ':>8}  {'CV5 σ':>8}"
    if test_agg:
        header += f"  {'Test μ':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for c in FG_CLASSES:
        k = f"dice_class_{c}"
        row = f"  {CLASS_NAMES[c]:<10}"
        per_fold_means: List[float] = []
        for f in folds_available:
            vals = cv5_agg[f].get(k, [])
            mu   = float(np.mean(vals)) if vals else float("nan")
            per_fold_means.append(mu)
            row += f"  {_fmt(mu):>8}"
        # CV5 global (toutes les valeurs patients de tous les folds)
        all_vals = [v for f in folds_available for v in cv5_agg[f].get(k, [])]
        cv5_mu   = float(np.mean(all_vals)) if all_vals else float("nan")
        cv5_sig  = float(np.std(all_vals))  if len(all_vals) > 1 else 0.0
        row += f"  {_fmt(cv5_mu):>8}  {_fmt(cv5_sig):>8}"
        if test_agg:
            tv = test_agg.get(k, [])
            row += f"  {_fmt(float(np.mean(tv)) if tv else float('nan')):>8}"
        print(row)

    # FG mean
    row = f"  {'FG mean':<10}"
    for f in folds_available:
        vals = cv5_agg[f].get("mean_dice_fg", [])
        mu   = float(np.mean(vals)) if vals else float("nan")
        row += f"  {_fmt(mu):>8}"
    all_fg = [v for f in folds_available for v in cv5_agg[f].get("mean_dice_fg", [])]
    cv5_fg_mu  = float(np.mean(all_fg)) if all_fg else float("nan")
    cv5_fg_sig = float(np.std(all_fg))  if len(all_fg) > 1 else 0.0
    row += f"  {_fmt(cv5_fg_mu):>8}  {_fmt(cv5_fg_sig):>8}"
    if test_agg:
        tv = test_agg.get("mean_dice_fg", [])
        row += f"  {_fmt(float(np.mean(tv)) if tv else float('nan')):>8}"
    print(row)


def print_verdict_table(
    cv5_agg: Dict[str, Dict[str, List[float]]],
    test_agg: Optional[Dict[str, List[float]]],
) -> None:
    """Tableau de verdict global sur les métriques CV5 agrégées."""
    print(f"\n{HEADER_LINE}")
    print("VERDICT GLOBAL CV5 (toutes patients confondus)")
    print(HEADER_LINE)
    print(f"  {'Classe':<10} {'Dice μ':>8} {'Dice σ':>8} {'Rec μ':>8}  Verdict")
    print("  " + "-" * 60)

    for c in FG_CLASSES:
        dk = f"dice_class_{c}"
        rk = f"recall_class_{c}"
        all_d = [v for agg in cv5_agg.values() for v in agg.get(dk, [])]
        all_r = [v for agg in cv5_agg.values() for v in agg.get(rk, [])]
        dm = float(np.mean(all_d)) if all_d else float("nan")
        ds = float(np.std(all_d))  if len(all_d) > 1 else 0.0
        rm = float(np.mean(all_r)) if all_r else float("nan")
        v  = _verdict(dm, ds, rm)
        print(f"  {CLASS_NAMES[c]:<10} {_fmt(dm):>8} {_fmt(ds):>8} {_fmt(rm):>8}  {v}")

    if test_agg:
        print(f"\n  {'[TEST]':<10} {'Dice μ':>8} {'σ':>8} {'Rec μ':>8}")
        for c in FG_CLASSES:
            dk = f"dice_class_{c}"
            rk = f"recall_class_{c}"
            td = test_agg.get(dk, [])
            tr = test_agg.get(rk, [])
            dm = float(np.mean(td)) if td else float("nan")
            ds = float(np.std(td))  if len(td) > 1 else 0.0
            rm = float(np.mean(tr)) if tr else float("nan")
            print(f"  {CLASS_NAMES[c]:<10} {_fmt(dm):>8} {_fmt(ds):>8} {_fmt(rm):>8}")
        fg = test_agg.get("mean_dice_fg", [])
        print(f"  {'FG mean':<10} {_fmt(float(np.mean(fg)) if fg else float('nan')):>8}")


def print_latex_table(
    cv5_agg: Dict[str, Dict[str, List[float]]],
    test_agg: Optional[Dict[str, List[float]]],
) -> None:
    """Génère un tableau LaTeX directement copiable dans la thèse."""
    folds = sorted(cv5_agg.keys())
    print(f"\n{HEADER_LINE}")
    print("TABLEAU LaTeX (copier-coller dans la thèse)")
    print(HEADER_LINE)
    cols = "l" + "r" * len(folds) + "rr"
    if test_agg:
        cols += "r"
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Dice scores per family class — 5-fold cross-validation, Stage-2 Level-1}")
    print(r"\label{tab:cv5_stage2_level1}")
    print(r"\begin{tabular}{" + cols + r"}")
    print(r"\toprule")

    # En-tête
    hdr = "Classe"
    for f in folds:
        hdr += f" & {f.replace('_', '\\_')}"
    hdr += r" & CV5 $\mu$ & CV5 $\sigma$"
    if test_agg:
        hdr += r" & Test $\mu$"
    hdr += r" \\"
    print(hdr)
    print(r"\midrule")

    for c in FG_CLASSES:
        k  = f"dice_class_{c}"
        row = CLASS_NAMES[c]
        per_fold: List[float] = []
        for f in folds:
            vals = cv5_agg[f].get(k, [])
            mu   = float(np.mean(vals)) if vals else float("nan")
            per_fold.append(mu)
            row += f" & {mu:.3f}"
        all_v   = [v for f in folds for v in cv5_agg[f].get(k, [])]
        cv5_mu  = float(np.mean(all_v)) if all_v else float("nan")
        cv5_sig = float(np.std(all_v))  if len(all_v) > 1 else 0.0
        row += f" & \\textbf{{{cv5_mu:.3f}}} & {cv5_sig:.3f}"
        if test_agg:
            tv  = test_agg.get(k, [])
            row += f" & {float(np.mean(tv)):.3f}" if tv else " & ---"
        row += r" \\"
        print(row)

    print(r"\midrule")
    row = r"\textbf{FG mean}"
    for f in folds:
        vals = cv5_agg[f].get("mean_dice_fg", [])
        mu   = float(np.mean(vals)) if vals else float("nan")
        row += f" & {mu:.3f}"
    all_fg   = [v for f in folds for v in cv5_agg[f].get("mean_dice_fg", [])]
    fg_mu    = float(np.mean(all_fg)) if all_fg else float("nan")
    fg_sig   = float(np.std(all_fg))  if len(all_fg) > 1 else 0.0
    row += f" & \\textbf{{{fg_mu:.3f}}} & {fg_sig:.3f}"
    if test_agg:
        tv   = test_agg.get("mean_dice_fg", [])
        row += f" & \\textbf{{{float(np.mean(tv)):.3f}}}" if tv else " & ---"
    row += r" \\"
    print(row)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agrège les résultats CV5 stage-2 Level-1")
    p.add_argument("--output-json", default="", help="Sauvegarde le résumé CV5 en JSON")
    p.add_argument("--latex", action="store_true", help="Génère un tableau LaTeX")
    p.add_argument("--no-test", action="store_true", help="Ignorer le fichier holdout test")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\n{HEADER_LINE}")
    print("AGRÉGATION CV5 — Stage-2 Level-1 (5 familles vasculaires)")
    print(HEADER_LINE)

    # Chargement des folds disponibles
    cv5_agg: Dict[str, Dict[str, List[float]]] = {}
    all_patients: List[Dict] = []

    for fold, path in sorted(FOLD_JSON_MAP.items()):
        report = load_fold_report(path)
        if report is None:
            print(f"  [{fold}] ⚠  Fichier absent : {path.relative_to(ROOT)}")
            continue
        patients = extract_per_patient_metrics(report)
        cv5_agg[fold] = aggregate_patients(patients)
        all_patients.extend(patients)
        print(f"  [{fold}] ✓  {len(patients)} patients  "
              f"(ckpt epoch={report.get('checkpoint_epoch', '?')})")

    if not cv5_agg:
        print("\n[erreur] Aucun fichier de résultats trouvé.")
        print("Vérifie que run_cv5.ps1 a été exécuté et que les JSONs existent.")
        return

    n_folds   = len(cv5_agg)
    n_patients = len(all_patients)
    print(f"\n  Folds chargés : {n_folds}/5   |   Patients totaux : {n_patients}/20")

    # Holdout test
    test_agg: Optional[Dict[str, List[float]]] = None
    if not args.no_test and TEST_JSON.exists():
        test_report  = load_fold_report(TEST_JSON)
        test_patients = extract_per_patient_metrics(test_report)
        test_agg     = aggregate_patients(test_patients)
        print(f"  [holdout test] ✓  {len(test_patients)} patients")
    elif not args.no_test:
        print(f"  [holdout test] ⚠  Absent : {TEST_JSON.name}")

    # Affichage par fold
    print(f"\n{HEADER_LINE}")
    print("DÉTAIL PAR FOLD")
    print(HEADER_LINE)
    for fold in sorted(cv5_agg.keys()):
        # Reconstituer la liste patients depuis le JSON original
        report   = load_fold_report(FOLD_JSON_MAP[fold])
        patients = extract_per_patient_metrics(report) if report else []
        print_fold_summary(fold, patients)

    # Tableau CV5 global
    print_cv5_table(cv5_agg, test_agg)
    print_verdict_table(cv5_agg, test_agg)

    if args.latex:
        print_latex_table(cv5_agg, test_agg)

    # Calcul métrique headline CV5
    all_fg = [v for agg in cv5_agg.values() for v in agg.get("mean_dice_fg", [])]
    if all_fg:
        print(f"\n{HEADER_LINE}")
        print("MÉTRIQUE HEADLINE (à reporter dans la thèse)")
        print(HEADER_LINE)
        print(f"  Stage-2 Level-1  —  {n_folds}-fold CV  —  {n_patients} patients\n")
        print(f"  mean_dice_FG = {np.mean(all_fg):.3f} ± {np.std(all_fg):.3f}")
        for c in FG_CLASSES:
            k    = f"dice_class_{c}"
            vals = [v for agg in cv5_agg.values() for v in agg.get(k, [])]
            mu   = float(np.mean(vals)) if vals else float("nan")
            sig  = float(np.std(vals))  if len(vals) > 1 else 0.0
            print(f"  {CLASS_NAMES[c]:<10} Dice = {mu:.3f} ± {sig:.3f}")
        if test_agg:
            tv = test_agg.get("mean_dice_fg", [])
            print(f"\n  [Holdout Test, n=5] mean_dice_FG = {np.mean(tv):.3f} ± {np.std(tv):.3f}")

    # Sauvegarde JSON
    if args.output_json:
        # Construire le résumé
        summary = {
            "n_folds": n_folds,
            "n_cv_patients": n_patients,
            "folds_available": sorted(cv5_agg.keys()),
            "class_names": CLASS_NAMES,
            "cv5_per_fold": {},
            "cv5_global": {},
            "holdout_test": {},
        }
        for fold, agg in cv5_agg.items():
            summary["cv5_per_fold"][fold] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v) if len(v) > 1 else 0.0)}
                for k, v in agg.items()
            }
        all_agg_global = aggregate_patients(all_patients)
        summary["cv5_global"] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v) if len(v) > 1 else 0.0)}
            for k, v in all_agg_global.items()
        }
        if test_agg:
            summary["holdout_test"] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v) if len(v) > 1 else 0.0)}
                for k, v in test_agg.items()
            }
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[done] Résumé CV5 sauvegardé → {out}")


if __name__ == "__main__":
    main()
