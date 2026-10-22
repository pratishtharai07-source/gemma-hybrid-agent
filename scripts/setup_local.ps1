<#
.SYNOPSIS
    Start the local Gemma 4 E2B tier (llama.cpp) for this project.

.DESCRIPTION
    Downloads the open-weight Gemma 4 E2B GGUF if it is missing, then starts
    llama-server on 127.0.0.1:8080 with flags tuned for a small machine
    (4 GB RAM): a small context, no mmap thrash, no warmup cost.

    The local tier is what makes this an open-weight project rather than a
    wrapper around a hosted endpoint.  It is entirely optional -- the app
    falls back to the Gemini API when this server is not running.

.EXAMPLE
    .\scripts\setup_local.ps1
    .\scripts\setup_local.ps1 -Port 8081 -ContextSize 4096
#>
[CmdletBinding()]
param(
    [int]$Port = 8080,
    [int]$ContextSize = 2048,
    [int]$Threads = 0,
    [switch]$DownloadOnly,
    [switch]$SkipDownload
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$ModelDir   = Join-Path $ProjectRoot 'models'
$ModelPath  = Join-Path $ModelDir 'gemma-4-E2B-it-Q3_K_M.gguf'
$ModelUrl   = 'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q3_K_M.gguf'
$ServerPath = Join-Path $ProjectRoot 'tools\llamacpp\llama-server.exe'
$ExpectedBytes = 2536000000

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "    $msg" -ForegroundColor Yellow }

if (-not (Test-Path $ModelDir)) { New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null }

# --- 1. model ------------------------------------------------------------
if (-not $SkipDownload) {
    $have = if (Test-Path $ModelPath) { (Get-Item $ModelPath).Length } else { 0 }
    if ($have -ge $ExpectedBytes) {
        Write-Ok "Model already present ($([math]::Round($have/1MB)) MB)."
    } else {
        Write-Step "Downloading Gemma 4 E2B Q3_K_M (~2.4 GB). This resumes if interrupted."
        $attempt = 0
        while ($attempt -lt 40) {
            $attempt++
            $before = if (Test-Path $ModelPath) { (Get-Item $ModelPath).Length } else { 0 }
            & curl.exe -L --retry 8 --retry-all-errors --retry-delay 4 -C - -o $ModelPath $ModelUrl 2>$null | Out-Null
            $after = if (Test-Path $ModelPath) { (Get-Item $ModelPath).Length } else { 0 }
            Write-Host ("    attempt {0}: {1} MB" -f $attempt, [math]::Round($after/1MB))
            if ($after -ge $ExpectedBytes) { break }
            if ($after -eq $before) { Start-Sleep -Seconds 5 }
        }
        if ((Get-Item $ModelPath).Length -lt $ExpectedBytes) {
            throw "Model download did not complete. Re-run this script to resume."
        }
        Write-Ok "Model downloaded."
    }
}

if ($DownloadOnly) { return }

# --- 2. runtime ----------------------------------------------------------
if (-not (Test-Path $ServerPath)) {
    throw "llama-server.exe not found at $ServerPath. Extract llama-b*-bin-win-cpu-x64.zip into tools\llamacpp\."
}
Write-Ok "Runtime: $ServerPath"

# --- 3. already running? ------------------------------------------------
try {
    $existing = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 3
    Write-Ok "A llama-server is already listening on port $Port."
    $existing.data | ForEach-Object { Write-Host "      model id: $($_.id)" -ForegroundColor DarkGray }
    return
} catch { }

# --- 4. start ------------------------------------------------------------
if ($Threads -le 0) {
    $Threads = [Environment]::ProcessorCount
    if ($Threads -gt 4) { $Threads = 4 }
}

$args = @(
    '-m', $ModelPath,
    '-c', $ContextSize,
    '-t', $Threads,
    '--host', '127.0.0.1',
    '--port', $Port,
    '--alias', 'gemma4-e2b',
    '--no-warmup'
)

Write-Step "Starting llama-server on 127.0.0.1:$Port (threads=$Threads, ctx=$ContextSize)"
$proc = Start-Process -FilePath $ServerPath -ArgumentList $args -PassThru -WindowStyle Minimized

# --- 5. wait for health --------------------------------------------------
$ready = $false
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { throw "llama-server exited with code $($proc.ExitCode)." }
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 3
        $ready = $true
        break
    } catch { }
}

if ($ready) {
    Write-Ok "Local Gemma 4 E2B is serving on http://127.0.0.1:$Port"
    Write-Host ""
    Write-Host "    PID: $($proc.Id)   (stop it with: Stop-Process -Id $($proc.Id))" -ForegroundColor DarkGray
    Write-Host "    The app auto-detects this server. Nothing else to configure." -ForegroundColor DarkGray
} else {
    Write-Warn2 "The server did not become healthy in time. Check the llama-server window."
}
