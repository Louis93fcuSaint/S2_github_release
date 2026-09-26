param(
    [int]$ChunkSize = 2000,
    [int]$Workers = 20,
    [int]$NumModes = 48
)
$ErrorActionPreference = "Stop"
$Root = "D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S1\04_versions\v4.7"
$Python = "C:\Users\fcu15\ross230_py312\python.exe"
$PyRoot = "C:\Users\fcu15\ross230_py312"
$Relabel = Join-Path $Root "relabel_old_v4_7.py"
$Combine = Join-Path $Root "combine_relabel_chunks_v4_7.py"

$env:PATH = "$PyRoot\Library\bin;$PyRoot\Scripts;$PyRoot;" + $env:PATH
$env:ROSS_FAST_RELABEL = "1"
$env:ROSS_ENABLE_NUMBA_JIT = "0"
$env:PYTHONWARNINGS = "ignore"

$Jobs = @(
    @{
        Name     = "v4.4_100k"
        Features = "D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S1\02_datasets\output_v4.4\merged_100k\features_v4.4_100000.csv"
        Dataset  = "D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S1\02_datasets\output_v4.4\merged_100k\dataset_v4.4_100000.csv"
        OutDir   = Join-Path $Root "relabeled_full_v4.7"
    },
    @{
        Name     = "v4.5_100k"
        Features = "D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S1\02_datasets\rotor2026-9\rotor2026-9\output_100k\features_v4.5_100000.csv"
        Dataset  = "D:\Louis_Projet\rotor_projet\大学生创新创业训练计划\S1\02_datasets\rotor2026-9\rotor2026-9\output_100k\dataset_v4.5_100000.csv"
        OutDir   = Join-Path $Root "relabeled_full_v4.7_v45"
    }
)

$LogRoot = Join-Path $Root "relabel_logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $LogRoot "relabel_v4.7_$Stamp.log"

function Write-Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    Write-Host $line
    Add-Content -LiteralPath $Log -Value $line -Encoding UTF8
}

Write-Log "START v4.7 relabel. chunk=$ChunkSize workers=$Workers num_modes=$NumModes"
Write-Log "python: $Python  log: $Log"

foreach ($job in $Jobs) {
    $OutDir = $job.OutDir
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $Total = (Get-Content -LiteralPath $job.Features | Measure-Object -Line).Lines - 1
    Write-Log "=== job $($job.Name): total=$Total out=$OutDir"

    for ($Start = 0; $Start -lt $Total; $Start += $ChunkSize) {
        $Limit = [Math]::Min($ChunkSize, $Total - $Start)
        $Stem = "${Start}_${Limit}"
        $Summary = Join-Path $OutDir "relabel_summary_v4.7_${Stem}.json"
        $Labels = Join-Path $OutDir "relabeled_v4.7_${Stem}.csv"
        $Modes = Join-Path $OutDir "relabel_modes_v4.7_${Stem}.csv"
        $Audit = Join-Path $OutDir "relabel_audit_v4.7_${Stem}.csv"

        if ((Test-Path $Summary) -and (Test-Path $Labels) -and
            (Test-Path $Modes) -and (Test-Path $Audit)) {
            try {
                $Saved = Get-Content -Raw -Encoding UTF8 $Summary | ConvertFrom-Json
                if ([int]$Saved.n_selected -eq $Limit -and [int]$Saved.num_modes -eq $NumModes) {
                    Write-Log "[skip] $($job.Name) chunk $Stem"
                    continue
                }
            } catch { }
        }

        Write-Log "[run ] $($job.Name) chunk $Stem"
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $Python -u $Relabel `
            --features $job.Features `
            --dataset $job.Dataset `
            --start $Start `
            --limit $Limit `
            --workers $Workers `
            --num-modes $NumModes `
            -o $OutDir *>> $Log
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $prev
        if ($exitCode -ne 0) {
            Write-Log "[FAIL] $($job.Name) chunk $Stem exit=$exitCode"
            throw "Chunk $Stem failed"
        }
        Write-Log "[ok  ] $($job.Name) chunk $Stem"
    }

    Write-Log "[combine] $($job.Name)"
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python -u $Combine --out-dir $OutDir --total $Total *>> $Log
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($exitCode -ne 0) { Write-Log "[FAIL] combine $($job.Name)"; throw "combine failed" }
    Write-Log "[done] $($job.Name)"
}

Write-Log "ALL DONE"