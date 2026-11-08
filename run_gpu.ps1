# One-command runner for the full paper N-BEATS grid on Windows (PowerShell). Replicates Adler et al.,
# arXiv:2510.16060v2. [Paper App. A.2, Table 3] 64 configs x 3 seeds x 2 selection readings x 3 datasets.
# Usage:  powershell -ExecutionPolicy Bypass -File run_gpu.ps1      (rerun the same command to resume)
# NOTE: written to mirror run_gpu.sh but not tested on Windows (the project box is Linux).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Root = (Get-Location).Path
New-Item -ItemType Directory -Force results\runlogs, results\forecasts | Out-Null
$VPY = Join-Path $Root ".venv-nbeats\Scripts\python.exe"

if (-not (Test-Path $VPY)) {
  $made = $false
  foreach ($v in @("3.11", "3.12")) {
    try { & py -$v -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { & py -$v -m venv .venv-nbeats; $made = $true; break } } catch {}
  }
  if (-not $made) { Write-Error "Python 3.11 or 3.12 not found. Install it from python.org (tick 'py launcher') and rerun." }
}
if (-not (Test-Path ".venv-nbeats\.installed")) {
  & $VPY -m pip install --upgrade pip
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & $VPY -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  } else {
    & $VPY -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
  }
  & $VPY -m pip install -r envs\requirements-nbeats-gpu.txt
  New-Item .venv-nbeats\.installed -ItemType File | Out-Null
}
$Dev = (& $VPY -c "import sys; sys.path.insert(0,'src'); import run_nbeats as R; print(R.pick_device('auto'))" | Select-Object -Last 1)
& {
  "date: $(Get-Date)"; "os: $([Environment]::OSVersion.VersionString)"; "device: $Dev"
  & $VPY -c "import platform, torch, neuralforecast; print('python', platform.python_version()); print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available()); print('neuralforecast', neuralforecast.__version__)"
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi } else { "nvidia-smi: not available" }
} *> results\runlogs\gpu_env.txt
Write-Host "=== Device: $Dev"
if ($Dev -eq "cpu") { Write-Warning "No CUDA GPU detected: CPU will take ~2 days. Ctrl-C to abort (continuing in 15 s)."; Start-Sleep 15 }

$ErrorActionPreference = "Continue"
if (-not (Test-Path results\forecasts\heartrate__nbeats-full-seed0.npz)) {
  Write-Host "=== Timing trial (configs 32,63 on Heart-Rate)"
  & $VPY src\run_nbeats.py --dataset heartrate --grid full --seed 0 --select insample --device $Dev --config-ids 32,63
}
$missing = 0
foreach ($seed in 0,1,2) { foreach ($sel in "insample","holdout") { foreach ($ds in "heartrate","crime","glucose") {
  $tag = "nbeats" + $(if ($sel -eq "holdout") {"-holdout"} else {""}) + "-full-seed$seed"
  if (Test-Path "results\forecasts\${ds}__$tag.npz") { "skip $ds $tag"; continue }
  "START $ds $tag $(Get-Date)" | Tee-Object -Append results\runlogs_nbeats_full.txt
  & $VPY src\run_nbeats.py --dataset $ds --select $sel --seed $seed --grid full --device $Dev 2>&1 | Tee-Object -Append results\runlogs_nbeats_full.txt
  if (-not (Test-Path "results\forecasts\${ds}__$tag.npz")) { $missing++ }
}}}
if ($missing -eq 0) { "FULL_GRID_DONE" } else { "INCOMPLETE: $missing runs missing -- rerun the same command to resume" }
$out = Join-Path (Split-Path $Root -Parent) "RETURN_ME.zip"
Get-ChildItem -Force $Root | Where-Object { $_.Name -notin @(".venv-nbeats", ".git") } | Compress-Archive -DestinationPath $out -Force
Write-Host "Wrote $out. Send this file back."
