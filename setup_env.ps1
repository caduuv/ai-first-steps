# setup_env.ps1 — Windows Environment Setup for AI First Steps
# Run this script from the project root: .\setup_env.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  AI First Steps — Environment Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://python.org" -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] Found $pythonVersion" -ForegroundColor Green

# Create virtual environment
$venvPath = ".venv"
if (Test-Path $venvPath) {
    Write-Host "[INFO] Virtual environment already exists at $venvPath" -ForegroundColor Yellow
} else {
    Write-Host "[...] Creating virtual environment..." -ForegroundColor Cyan
    python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
}

# Activate virtual environment
Write-Host "[...] Activating virtual environment..." -ForegroundColor Cyan
& "$venvPath\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "[...] Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

# Install dependencies
Write-Host "[...] Installing dependencies (this may take a few minutes)..." -ForegroundColor Cyan
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install dependencies." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment in the future, run:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "Start learning with:" -ForegroundColor Cyan
Write-Host "  Open module_01_fundamentals\README.md" -ForegroundColor White
Write-Host ""
