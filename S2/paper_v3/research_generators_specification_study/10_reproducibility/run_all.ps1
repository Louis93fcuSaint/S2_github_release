param(
    [switch]$Full,
    [string]$Python = "D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S2\neural_surrogate_experiment\.venv_torch\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$logDir = Join-Path $here "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Step {
    param([string]$Name, [string]$Script, [string]$WorkDir)
    Write-Host "=== $Name ==="
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $log = Join-Path $logDir ("{0}_{1}.log" -f $Name, $stamp)
    Push-Location $WorkDir
    try {
        & $Python $Script 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
    }
    finally {
        Pop-Location
    }
}

if ($Full) {
    Invoke-Step -Name "00_inventory" -Script "build_inventory.py" -WorkDir $here
}
Invoke-Step -Name "01_spec_difficulty" -Script "build_spec_difficulty.py" -WorkDir $here
Invoke-Step -Name "02_reaggregate" -Script "reaggregate_four_way.py" -WorkDir $here
Invoke-Step -Name "03_cost_ci" -Script "build_cost_ci.py" -WorkDir $here
Invoke-Step -Name "04_diversity" -Script "analyze_diversity.py" -WorkDir $here
Invoke-Step -Name "05_screening" -Script "analyze_screening.py" -WorkDir $here
Invoke-Step -Name "06_difficulty" -Script "analyze_difficulty.py" -WorkDir $here
Invoke-Step -Name "07_retrieval" -Script "analyze_retrieval.py" -WorkDir $here
Invoke-Step -Name "08_figures" -Script (Join-Path $root "08_figures\scripts\make_figures.py") -WorkDir $root

Write-Host ""
Write-Host "All stages completed. Logs: $logDir"
