param(
    [string]$Python = "py",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = Join-Path $projectRoot "agenticrag"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -3.11 -m venv $venvPath
}
if (-not $SkipInstall) {
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -e "${projectRoot}[cli,dev]" "pyinstaller>=6.10,<7"
}
$pytestTemp = Join-Path $projectRoot ".pytest-tmp-build-script"
& $venvPython -m pytest (Join-Path $projectRoot "tests") -q --basetemp $pytestTemp
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed with exit code $LASTEXITCODE"
}
& $venvPython -m PyInstaller --noconfirm --clean (Join-Path $projectRoot "packaging\AutoMemory.spec")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$exePath = Join-Path $projectRoot "dist\AutoMemory.exe"
if (-not (Test-Path -LiteralPath $exePath)) {
    throw "Build completed without producing $exePath"
}
& $exePath --version
if ($LASTEXITCODE -ne 0) {
    throw "AutoMemory smoke test failed with exit code $LASTEXITCODE"
}
Write-Host "Built: $exePath"
