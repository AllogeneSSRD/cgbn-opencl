param(
    [string]$N_expr  = "2^4027-1", # 421 1009 2017 3049 4027
    [string]$B1_list = "1e3" 				# 1e4 1e5
)

$curves = @(1, 32, 64, 128, 256, 384, 512, 1024, 1536, 2048, 3072, 4096, 6144, 9216, 12288, 16384)
#1, 32, 64, 128, 256, 384, 512, 1024, 1536, 2048, 3072, 4096, 6144, 9216, 12288, 16384

$exe    = Join-Path $PSScriptRoot "build_rel\Release\ecm.exe"
$sigma  = "2026"
$device = 1

if (-not (Test-Path $exe)) {
    Write-Host "[ERROR] $exe not found"
    Write-Host "Build first: cmake --build build --config Debug"
    Pause
    exit 1
}

$logDir  = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logDir)) { $null = mkdir $logDir }

$ts      = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logDir "bench_$ts.log"
$tmpFile = Join-Path $logDir "_tmp_$ts.txt"

Write-Host ""
Write-Host "============================================================"
Write-Host " ECM GPU Stage1 Benchmark"
Write-Host " Root   = $PSScriptRoot\"
Write-Host " N      = $N_expr"
Write-Host " B1     = $B1_list"
Write-Host " Device = $device"
Write-Host " Log    = $logFile"
Write-Host "============================================================"
Write-Host ""

# --- log header ---
@"
================================================================
 ECM GPU Stage1 Benchmark Log
 Time   = $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
 N      = $N_expr
 B1     = $B1_list
 Device = $device
================================================================

"@ | Out-File -FilePath $logFile -Encoding ASCII

# --- main loop ---
foreach ($b1 in $B1_list -split ' ') {
    if (-not $b1) { continue }
    Write-Host ""
    Write-Host "========== B1=$b1 =========="

    foreach ($c in $curves) {
        $tag = "[curves=$c  B1=$b1]"
        Write-Host "$tag warmup..."

        # warmup
        # "---- warmup  curves=$c  B1=$b1 ----" | Out-File -FilePath $logFile -Append -Encoding ASCII
        # echo "($N_expr)" | & $exe -v -d $device -gpu -sigma 3:$sigma -gpucurves $c $b1 0 *>&1 |
        #     Out-File -FilePath $logFile -Append -Encoding ASCII
        # "" | Out-File -FilePath $logFile -Append -Encoding ASCII

        # --- run 1 ---
        "---- run 1  curves=$c  B1=$b1 ----" | Out-File -FilePath $logFile -Append -Encoding ASCII
        echo "($N_expr)" | & $exe -v -d $device -gpu -sigma 3:$sigma -gpucurves $c $b1 0 *>&1 |
            Tee-Object -FilePath $tmpFile |
            Out-File -FilePath $logFile -Append -Encoding ASCII

        Get-Content $tmpFile | Select-String "Using |gputime=" | ForEach-Object {
            Write-Host "  $_"
        }
        $line  = Get-Content $tmpFile | Select-String "gputime=" | Select-Object -First 1
        $gt1   = if ($line) { ($line -split "gputime=")[1] } else { "N/A" }

        # --- run 2 ---
        "---- run 2  curves=$c  B1=$b1 ----" | Out-File -FilePath $logFile -Append -Encoding ASCII
        echo "($N_expr)" | & $exe -v -d $device -gpu -sigma 3:$sigma -gpucurves $c $b1 0 *>&1 |
            Tee-Object -FilePath $tmpFile |
            Out-File -FilePath $logFile -Append -Encoding ASCII

        Get-Content $tmpFile | Select-String "Using |gputime=" | ForEach-Object {
            Write-Host "  $_"
        }
        $line2 = Get-Content $tmpFile | Select-String "gputime=" | Select-Object -First 1
        $gt2   = if ($line2) { ($line2 -split "gputime=")[1] } else { "N/A" }

        Write-Host "  $tag  run1: $gt1  run2: $gt2"
        "" | Out-File -FilePath $logFile -Append -Encoding ASCII
    }
}

Remove-Item $tmpFile -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "============================================================"
Write-Host " Benchmark complete."
Write-Host " Log: $logFile"
Write-Host "============================================================"
