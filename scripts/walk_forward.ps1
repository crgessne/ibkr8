# Walk-forward validation for NN model
# 6 folds: 4-year rolling train, 1-year test
# Each fold retrains a fresh model — no lookahead

$env:PYTHONIOENCODING = "utf-8"
$python = "C:\Users\Administrator\ibkr8\.venv\Scripts\python.exe"
$script = "C:\Users\Administrator\ibkr8\scripts\master_pipeline.py"
$outDir = "C:\Users\Administrator\ibkr8"

# Define folds: train_start-train_end, test_start-test_end
$folds = @(
    @{ train = "2016-2019"; test = "2020-2020"; label = "fold1_test2020" },
    @{ train = "2017-2020"; test = "2021-2021"; label = "fold2_test2021" },
    @{ train = "2018-2021"; test = "2022-2022"; label = "fold3_test2022" },
    @{ train = "2019-2022"; test = "2023-2023"; label = "fold4_test2023" },
    @{ train = "2020-2023"; test = "2024-2024"; label = "fold5_test2024" },
    @{ train = "2021-2024"; test = "2025-2025"; label = "fold6_test2025" }
)

$totalFolds = $folds.Count
$startTime = Get-Date

Write-Host ""
Write-Host "============================================================"
Write-Host " WALK-FORWARD VALIDATION: NN + setup_filter + prob_weighted"
Write-Host " $totalFolds folds, 4-year rolling train, 1-year test"
Write-Host " Started: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "============================================================"
Write-Host ""

$foldNum = 0
foreach ($fold in $folds) {
    $foldNum++
    $label = $fold.label
    $trainYears = $fold.train
    $testYears = $fold.test
    $outFile = "$outDir\_wf_nn_$label.txt"

    $foldStart = Get-Date
    Write-Host "============================================================"
    Write-Host " FOLD $foldNum / $totalFolds : $label"
    Write-Host " Train: $trainYears  |  Test: $testYears"
    Write-Host " Output: $outFile"
    Write-Host " Fold started: $($foldStart.ToString('HH:mm:ss'))"
    Write-Host "============================================================"

    & $python $script `
        --model-kind nn `
        --setup-filter `
        --select-mode prob_weighted `
        --train-years $trainYears `
        --test-years $testYears `
        2>&1 | Tee-Object -FilePath $outFile

    $foldEnd = Get-Date
    $foldElapsed = $foldEnd - $foldStart
    $totalElapsed = $foldEnd - $startTime

    Write-Host ""
    Write-Host "------------------------------------------------------------"
    Write-Host " FOLD $foldNum / $totalFolds DONE: $label"
    Write-Host " Fold time: $($foldElapsed.ToString('mm\:ss'))"
    Write-Host " Total elapsed: $($totalElapsed.ToString('hh\:mm\:ss'))"
    if ($foldNum -lt $totalFolds) {
        $avgPerFold = $totalElapsed.TotalSeconds / $foldNum
        $remaining = [TimeSpan]::FromSeconds($avgPerFold * ($totalFolds - $foldNum))
        Write-Host " Est. remaining: $($remaining.ToString('hh\:mm\:ss')) ($($totalFolds - $foldNum) folds left)"
    }
    Write-Host "------------------------------------------------------------"
    Write-Host ""
}

# ============================================================
# AGGREGATE SUMMARY
# ============================================================
$endTime = Get-Date
$totalTime = $endTime - $startTime

Write-Host ""
Write-Host "============================================================"
Write-Host " WALK-FORWARD COMPLETE"
Write-Host " Total time: $($totalTime.ToString('hh\:mm\:ss'))"
Write-Host " Finished: $($endTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Host "============================================================"
Write-Host ""
Write-Host "STOP SUMMARIES (all stops per fold):"
Write-Host ""

foreach ($fold in $folds) {
    $label = $fold.label
    $outFile = "$outDir\_wf_nn_$label.txt"
    Write-Host "=== $label (train=$($fold.train) test=$($fold.test)) ==="
    if (Test-Path $outFile) {
        Get-Content $outFile | Select-String -Pattern "STOP SUMMARY"
    } else {
        Write-Host "  [ERROR] Output file not found: $outFile"
    }
    Write-Host ""
}

Write-Host "============================================================"
Write-Host " VALUE-ADD TABLES:"
Write-Host "============================================================"
foreach ($fold in $folds) {
    $label = $fold.label
    $outFile = "$outDir\_wf_nn_$label.txt"
    Write-Host ""
    Write-Host "=== $label ==="
    if (Test-Path $outFile) {
        $inTable = $false
        Get-Content $outFile | ForEach-Object {
            if ($_ -match "RF MODEL VALUE-ADD") { $inTable = $true }
            if ($inTable) {
                Write-Host $_
                if ($_ -match "^\s*$" -and $inTable) { $inTable = $false }
            }
        }
    }
}
