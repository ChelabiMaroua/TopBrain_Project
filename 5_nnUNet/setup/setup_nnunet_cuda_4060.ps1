param(
    [string]$PythonCmd = "python",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu121",
    [switch]$SkipNnUNetInstall,
    [switch]$ForceReinstallTorch,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Write-Host "=== nnUNet CUDA setup (RTX 4060 profile) ==="
Write-Host "PythonCmd=$PythonCmd"
Write-Host "TorchIndexUrl=$TorchIndexUrl"

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "NVIDIA GPU detected (nvidia-smi available)."
}
else {
    Write-Host "Warning: nvidia-smi not found in PATH. CUDA driver may be missing on this machine."
}

$verifyCode = @'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
'@

if ($DryRun) {
    Write-Host "Dry run enabled. Planned commands:"
    Write-Host " - $PythonCmd -m pip install --upgrade pip setuptools wheel"
    if ($ForceReinstallTorch) {
        Write-Host " - $PythonCmd -m pip uninstall -y torch torchvision torchaudio"
    }
    Write-Host " - $PythonCmd -m pip install --upgrade torch torchvision torchaudio --index-url $TorchIndexUrl"
    if (-not $SkipNnUNetInstall) {
        Write-Host " - $PythonCmd -m pip install --upgrade nnunetv2"
    }
    Write-Host " - python verification script (torch/cuda checks)"
    exit 0
}

Write-Host "Running: $PythonCmd -m pip install --upgrade pip setuptools wheel"
& $PythonCmd -m pip install --upgrade pip setuptools wheel

if ($ForceReinstallTorch) {
    Write-Host "Running: $PythonCmd -m pip uninstall -y torch torchvision torchaudio"
    & $PythonCmd -m pip uninstall -y torch torchvision torchaudio
}

Write-Host "Running: $PythonCmd -m pip install --upgrade torch torchvision torchaudio --index-url $TorchIndexUrl"
& $PythonCmd -m pip install --upgrade torch torchvision torchaudio --index-url $TorchIndexUrl

if (-not $SkipNnUNetInstall) {
    Write-Host "Running: $PythonCmd -m pip install --upgrade nnunetv2"
    & $PythonCmd -m pip install --upgrade nnunetv2
}

$verifyFile = Join-Path $env:TEMP "nnunet_cuda_verify.py"
Set-Content -Path $verifyFile -Value $verifyCode -Encoding UTF8
Write-Host "Running: $PythonCmd $verifyFile"
& $PythonCmd $verifyFile
Remove-Item $verifyFile -ErrorAction SilentlyContinue

$cudaCheck = & $PythonCmd -c "import torch; print(int(torch.cuda.is_available()))"
if ($cudaCheck.Trim() -ne "1") {
    throw "PyTorch CUDA is not available after setup. Check NVIDIA driver/CUDA runtime and reinstall torch with a CUDA wheel."
}

Write-Host "CUDA setup complete. You can now run nnUNet training on GPU."
