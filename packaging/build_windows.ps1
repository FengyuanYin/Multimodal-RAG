param(
    [string]$Python = "py",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = Join-Path $projectRoot ".venv-build"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -3.11 -m venv $venvPath
}
if (-not $SkipInstall) {
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -e "${projectRoot}[cli,dev]" "pyinstaller>=6.10,<7"
}
& $venvPython -m pytest (Join-Path $projectRoot "tests") -q
& $venvPython -m PyInstaller --noconfirm --clean (Join-Path $projectRoot "packaging\AutoMemory.spec")

$exePath = Join-Path $projectRoot "dist\AutoMemory.exe"
if (-not (Test-Path -LiteralPath $exePath)) {
    throw "Build completed without producing $exePath"
}
& $exePath --version
Write-Host "Built: $exePath"
