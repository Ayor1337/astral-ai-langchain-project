param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = "Stop"

$script_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$project_root = Split-Path -Parent $script_dir

Set-Location $project_root

$conda_executable = (Get-Command conda.exe -ErrorAction Stop).Source

# 统一通过 astral_ai Conda 环境执行测试，避免依赖当前 shell 的激活状态。
# 使用 --no-capture-output 规避 Windows 下 conda 转发 pytest 输出时的编码问题。
& $conda_executable run --no-capture-output -n astral_ai python -m pytest @PytestArgs
exit $LASTEXITCODE
