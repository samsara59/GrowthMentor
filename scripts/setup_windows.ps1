
# GrowthMentor-Agent Windows installation

$ErrorActionPreference = "Stop"

# 使用脚本文件所在位置定位项目根目录
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

Write-Host "GrowthMentor-Agent Setup"
Write-Host "Project: $ROOT"

# 检查 Python
python --version

# 如果当前没有激活虚拟环境才新建 .venv
if (-not $env:CONDA_DEFAULT_ENV -and
    -not $env:VIRTUAL_ENV) {

    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        python -m venv .venv
    }

    $PY = ".\.venv\Scripts\python.exe"

} else {
    # 使用现有 Conda / venv 环境
    $PY = "python"

    Write-Host "Using existing Python environment"
}

Write-Host "Installing dependencies..."

& $PY -m pip install -r requirements.txt --index-url https://pypi.org/simple

if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed"
}

Write-Host "Installation complete."
Write-Host "Start application with:"
Write-Host "python -m streamlit run app/ui/streamlit_app.py"
