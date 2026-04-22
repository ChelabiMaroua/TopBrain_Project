#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Cross-validation 5-folds : Stage-1 (binaire) + Stage-2 (Level-1, 5 familles).
    
.DESCRIPTION
    Pour chaque fold (fold_2 à fold_5), le script exécute séquentiellement :
      1. Entraînement stage-1 SwinUNETR binaire
      2. Inférence stage-1 → masques binaires pour tous les patients
      3. Ingestion MongoDB Level-1 dans une collection fold-spécifique
      4. Entraînement stage-2 SwinUNETR 5-classes
      5. Évaluation sur le val set du fold → JSON de résultats

    fold_1 est supposé déjà terminé (checkpoint + diagnostic existants).
    Relancer ce script est idempotent : les checkpoints existants sont ignorés par
    --early-stopping, et --overwrite sur les collections MongoDB garantit la fraîcheur.

.NOTES
    Temps estimé : ~3-5h par fold sur RTX 3080 10GB.
    Total ~12-20h pour fold_2 à fold_5.
    
.PARAMETER Folds
    Liste des folds à entraîner. Défaut : fold_2 fold_3 fold_4 fold_5
    
.PARAMETER StartFold
    Reprendre à partir d'un fold spécifique (ex: fold_3 si fold_2 a planté).
    
.PARAMETER Stage1Only
    N'exécuter que l'entraînement stage-1 (sans stage-2).
    
.PARAMETER Stage2Only
    N'exécuter que l'entraînement stage-2 (stage-1 doit déjà être fait).
    Requiert que les checkpoints stage-1 et les collections Level-1 existent.

.EXAMPLE
    # Lancer CV complète fold_2 à fold_5
    .\run_cv5.ps1
    
.EXAMPLE
    # Reprendre à fold_3 si fold_2 a déjà tourné
    .\run_cv5.ps1 -StartFold fold_3
    
.EXAMPLE
    # Uniquement stage-2 (stage-1 déjà entraîné pour tous les folds)
    .\run_cv5.ps1 -Stage2Only
#>

[CmdletBinding()]
param(
    [string[]]$Folds       = @("fold_2", "fold_3", "fold_4", "fold_5"),
    [string]  $StartFold   = "",
    [switch]  $Stage1Only,
    [switch]  $Stage2Only
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─── Configuration ────────────────────────────────────────────────────────────
$PYTHON       = "env_gpu/Scripts/python.exe"
$PARTITION    = "3_Data_Partitionement/partition_materialized.json"
$SRC_COLL     = "MultiClassPatients3D_Binary_CTA41"
$TARGET_SIZE  = "128x128x64"
$SWIN_FS      = 24
$PATCH        = @(64, 64, 64)
$STAGE1_DIR   = "4_Unet3D/checkpoints/stage1_binary_v2"
$STAGE2_DIR   = "5_HierarchicalSeg/checkpoints/stage2_level1_v1"
$RESULTS_DIR  = "results"

# Hyperparamètres stage-1 (identiques à fold_1)
$S1_EPOCHS       = 300
$S1_BATCH        = 1
$S1_ACCUM        = 8
$S1_PPV          = 12          # patches per volume
$S1_FG_OV        = 0.90
$S1_LR           = "3e-4"
$S1_CWEIGHTS     = "0.05,1.0"
$S1_EARLY_STOP   = 50
$S1_MAX_HOURS    = 12

# Hyperparamètres stage-2 (identiques à fold_1 — classe weights median-freq)
$S2_EPOCHS       = 300
$S2_BATCH        = 1
$S2_ACCUM        = 8
$S2_PPV          = 12
$S2_FG_OV        = 0.90
$S2_LR           = "3e-4"
$S2_CWEIGHTS     = "0.050,1.388,1.031,0.971,0.527"
$S2_EARLY_STOP   = 50
$S2_MAX_HOURS    = 12

# Seuil binaire pour l'inférence stage-1
$BIN_THRESHOLD   = 0.35

# ─── Helpers ──────────────────────────────────────────────────────────────────
function Write-Banner([string]$msg, [string]$color = "Cyan") {
    $line = "=" * 70
    Write-Host "`n$line" -ForegroundColor $color
    Write-Host "  $msg" -ForegroundColor $color
    Write-Host "$line`n" -ForegroundColor $color
}

function Assert-ExitCode([int]$code, [string]$step) {
    if ($code -ne 0) {
        Write-Host "[ERREUR] Étape '$step' a échoué (exit code $code)." -ForegroundColor Red
        exit $code
    }
}

function Test-Checkpoint([string]$path) {
    return Test-Path $path -PathType Leaf
}

# ─── Filtrage des folds selon --StartFold ─────────────────────────────────────
if ($StartFold -ne "") {
    $idx = $Folds.IndexOf($StartFold)
    if ($idx -lt 0) {
        Write-Error "-StartFold '$StartFold' n'est pas dans la liste : $($Folds -join ', ')"
        exit 1
    }
    $Folds = $Folds[$idx..($Folds.Count - 1)]
    Write-Host "[info] Reprise à partir de $StartFold → folds : $($Folds -join ', ')" -ForegroundColor Yellow
}

# ─── Boucle principale ────────────────────────────────────────────────────────
$allFoldsStart = Get-Date

foreach ($fold in $Folds) {
    $foldStart = Get-Date
    Write-Banner "FOLD : $fold" "Magenta"

    $stage1Ckpt    = "$STAGE1_DIR/swinunetr_best_${fold}.pth"
    $masksDir      = "$RESULTS_DIR/stage1_binary_masks_cv_${fold}"
    $manifestPath  = "$masksDir/stage1_inference_manifest.json"
    $level1Coll    = "HierarchicalPatients3D_Level1_CTA41_${fold}"
    $stage2Ckpt    = "$STAGE2_DIR/swinunetr_level1_best_${fold}.pth"
    $diagJson      = "$RESULTS_DIR/level1_diag_${fold}_val.json"
    $ckptName      = "swinunetr_best_${fold}_v2"

    # ── Étape 1 : Entraînement stage-1 ──────────────────────────────────────
    if (-not $Stage2Only) {
        if (Test-Checkpoint $stage1Ckpt) {
            Write-Host "[skip] Checkpoint stage-1 déjà présent : $stage1Ckpt" -ForegroundColor Yellow
        } else {
            Write-Banner "$fold | Étape 1/4 : Entraînement stage-1 binaire" "Cyan"
            & $PYTHON "4_Unet3D/train_unet3d_binary.py" `
                --collection      $SRC_COLL `
                --target-size     $TARGET_SIZE `
                --partition-file  $PARTITION `
                --num-classes     2 `
                --fold            $fold `
                --epochs          $S1_EPOCHS `
                --patch-size      $PATCH[0] $PATCH[1] $PATCH[2] `
                --swin-feature-size $SWIN_FS `
                --batch-size      $S1_BATCH `
                --accum-steps     $S1_ACCUM `
                --patches-per-volume $S1_PPV `
                --train-fg-oversample-prob $S1_FG_OV `
                --loss            dicece `
                --lambda-dice     2.0 `
                --lambda-ce       0.5 `
                --class-weights   $S1_CWEIGHTS `
                --lr              $S1_LR `
                --augment `
                --amp `
                --early-stopping  $S1_EARLY_STOP `
                --max-hours       $S1_MAX_HOURS `
                --save-dir        $STAGE1_DIR
            Assert-ExitCode $LASTEXITCODE "stage-1 training $fold"
        }

        # ── Étape 2 : Inférence stage-1 (tous les patients) ─────────────────
        if (Test-Path $manifestPath) {
            Write-Host "[skip] Manifest stage-1 déjà présent : $manifestPath" -ForegroundColor Yellow
        } else {
            Write-Banner "$fold | Étape 2/4 : Inférence stage-1 → masques binaires" "Cyan"
            & $PYTHON "5_HierarchicalSeg/level1_families/predict_stage1_from_mongo.py" `
                --checkpoint      $stage1Ckpt `
                --collection      $SRC_COLL `
                --target-size     $TARGET_SIZE `
                --num-classes     2 `
                --patch-size      $PATCH[0] $PATCH[1] $PATCH[2] `
                --swin-feature-size $SWIN_FS `
                --threshold       $BIN_THRESHOLD `
                --amp `
                --overwrite `
                --output-dir      $masksDir
            Assert-ExitCode $LASTEXITCODE "stage-1 inference $fold"
        }

        # ── Étape 3 : Ingestion Level-1 MongoDB ─────────────────────────────
        Write-Banner "$fold | Étape 3/4 : Ingestion Level-1 → $level1Coll" "Cyan"
        & $PYTHON "5_HierarchicalSeg/level1_families/ingest_level1_mongo.py" `
            --manifest            $manifestPath `
            --stage1-checkpoint-name $ckptName `
            --dst-collection      $level1Coll `
            --overwrite
        Assert-ExitCode $LASTEXITCODE "ingest level-1 $fold"
    }

    if ($Stage1Only) {
        Write-Host "[info] --Stage1Only : on s'arrête ici pour $fold" -ForegroundColor Yellow
        continue
    }

    # ── Étape 4 : Entraînement stage-2 ──────────────────────────────────────
    if (Test-Checkpoint $stage2Ckpt) {
        Write-Host "[skip] Checkpoint stage-2 déjà présent : $stage2Ckpt" -ForegroundColor Yellow
    } else {
        Write-Banner "$fold | Étape 4/4 : Entraînement stage-2 (5 familles)" "Cyan"
        & $PYTHON "5_HierarchicalSeg/level1_families/train_level1.py" `
            --collection          $level1Coll `
            --target-size         $TARGET_SIZE `
            --partition-file      $PARTITION `
            --num-classes         5 `
            --fold                $fold `
            --epochs              $S2_EPOCHS `
            --patch-size          $PATCH[0] $PATCH[1] $PATCH[2] `
            --swin-feature-size   $SWIN_FS `
            --batch-size          $S2_BATCH `
            --accum-steps         $S2_ACCUM `
            --patches-per-volume  $S2_PPV `
            --train-fg-oversample-prob $S2_FG_OV `
            --loss                dicece `
            --lambda-dice         2.0 `
            --lambda-ce           0.5 `
            --class-weights       $S2_CWEIGHTS `
            --lr                  $S2_LR `
            --augment `
            --amp `
            --early-stopping      $S2_EARLY_STOP `
            --max-hours           $S2_MAX_HOURS `
            --init-checkpoint     $stage1Ckpt `
            --log-label-distribution `
            --log-foreground-ratio `
            --save-dir            $STAGE2_DIR
        Assert-ExitCode $LASTEXITCODE "stage-2 training $fold"
    }

    # ── Évaluation val set ───────────────────────────────────────────────────
    Write-Banner "$fold | Évaluation sur val set" "Green"
    & $PYTHON "diagnose_level1_families.py" `
        --checkpoint      $stage2Ckpt `
        --collection      $level1Coll `
        --partition-file  $PARTITION `
        --fold            $fold `
        --split           val `
        --target-size     $TARGET_SIZE `
        --patch-size      $PATCH[0] $PATCH[1] $PATCH[2] `
        --swin-feature-size $SWIN_FS `
        --amp `
        --output-json     $diagJson
    Assert-ExitCode $LASTEXITCODE "evaluation $fold"

    $elapsed = (Get-Date) - $foldStart
    Write-Host "`n[done] $fold terminé en $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
}

# ─── Résumé final ─────────────────────────────────────────────────────────────
$totalElapsed = (Get-Date) - $allFoldsStart
Write-Banner "CV5 TERMINÉE — agrégation des résultats" "Green"
Write-Host "Durée totale : $($totalElapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host ""
Write-Host "Prochaine étape : lancer aggregate_cv5_results.py pour le tableau récapitulatif." -ForegroundColor Cyan
Write-Host "    env_gpu/Scripts/python.exe aggregate_cv5_results.py" -ForegroundColor White
Write-Host ""
Write-Host "Fichiers de résultats attendus :" -ForegroundColor Cyan
Write-Host "    results/level1_diag_fold_1.json    (fold_1, déjà fait)" -ForegroundColor White
foreach ($fold in @("fold_2", "fold_3", "fold_4", "fold_5")) {
    Write-Host "    results/level1_diag_${fold}_val.json" -ForegroundColor White
}
