[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ManifestPath,

    [string]$OutputPath,

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,

        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "path value must be non-empty"
    }

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }

    return Join-Path -Path $Root -ChildPath $Path
}

function Get-CommandArray {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Target,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $items = @($Target.command)
    if ($items.Count -eq 0) {
        throw "$Label command must contain at least one item"
    }

    return @($items | ForEach-Object { [string]$_ })
}

function New-OutputSummary {
    param([AllowNull()][string]$Stdout)

    $excerpt = $null
    if (-not [string]::IsNullOrWhiteSpace($Stdout)) {
        $excerpt = $Stdout.Trim()
        if ($excerpt.Length -gt 2000) {
            $excerpt = $excerpt.Substring(0, 2000)
        }
    }

    return [ordered]@{
        token_count = $null
        text = $null
        stdout_excerpt = $excerpt
    }
}

function New-RunRecord {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Command,

        [Parameter(Mandatory = $true)]
        [string]$ModelPath,

        [Parameter(Mandatory = $true)]
        [string]$AudioPath,

        [Parameter(Mandatory = $true)]
        [int]$Threads,

        [Parameter(Mandatory = $true)]
        [string]$Status,

        [AllowNull()][Nullable[int64]]$WallTimeMs,

        [AllowNull()][Nullable[int]]$ExitCode,

        [AllowNull()][string]$Stdout,

        [AllowNull()][object]$SkipReason
    )

    return [ordered]@{
        command = @($Command)
        model_path = $ModelPath
        audio_path = $AudioPath
        threads = $Threads
        status = $Status
        wall_time_ms = $WallTimeMs
        exit_code = $ExitCode
        output = New-OutputSummary -Stdout $Stdout
        skip_reason = $SkipReason
    }
}

function New-Record {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Manifest,

        [Parameter(Mandatory = $true)]
        [string]$Status,

        [Parameter(Mandatory = $true)]
        [object]$Ocelotl,

        [Parameter(Mandatory = $true)]
        [object]$WhisperCpp
    )

    return [ordered]@{
        fixture_version = 1
        manifest_name = [string]$Manifest.name
        status = $Status
        ocelotl = $Ocelotl
        whisper_cpp = $WhisperCpp
    }
}

function Invoke-BenchmarkCommand {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Command,

        [Parameter(Mandatory = $true)]
        [string]$ModelPath,

        [Parameter(Mandatory = $true)]
        [string]$AudioPath,

        [Parameter(Mandatory = $true)]
        [int]$Threads
    )

    $executable = [string]$Command[0]
    $arguments = @()
    if ($Command.Count -gt 1) {
        $arguments = @($Command[1..($Command.Count - 1)])
    }

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $output = & $executable @arguments 2>&1 | Out-String
        $exitCode = $LASTEXITCODE
        if ($null -eq $exitCode) {
            $exitCode = 0
        }
        $stopwatch.Stop()

        $status = if ($exitCode -eq 0) { "completed" } else { "failed" }
        return New-RunRecord `
            -Command $Command `
            -ModelPath $ModelPath `
            -AudioPath $AudioPath `
            -Threads $Threads `
            -Status $status `
            -WallTimeMs ([int64]$stopwatch.ElapsedMilliseconds) `
            -ExitCode ([int]$exitCode) `
            -Stdout $output `
            -SkipReason $null
    } catch {
        $stopwatch.Stop()
        return New-RunRecord `
            -Command $Command `
            -ModelPath $ModelPath `
            -AudioPath $AudioPath `
            -Threads $Threads `
            -Status "failed" `
            -WallTimeMs ([int64]$stopwatch.ElapsedMilliseconds) `
            -ExitCode 1 `
            -Stdout $_.Exception.Message `
            -SkipReason $null
    }
}

function Write-Record {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Record,

        [AllowNull()][string]$OutputPath
    )

    $json = $Record | ConvertTo-Json -Depth 12
    if (-not [string]::IsNullOrWhiteSpace($OutputPath)) {
        $parent = Split-Path -Parent $OutputPath
        if (-not [string]::IsNullOrWhiteSpace($parent)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        Set-Content -LiteralPath $OutputPath -Value $json -Encoding UTF8
    }
    Write-Output $json
}

$repoRoot = (Get-Location).Path
$manifestFullPath = Resolve-RepoPath -Root $repoRoot -Path $ManifestPath
$manifest = Get-Content -Raw -LiteralPath $manifestFullPath | ConvertFrom-Json

if ($manifest.fixture_version -ne 1) {
    throw "benchmark manifest fixture_version must be 1"
}
if ([int]$manifest.threads -lt 1) {
    throw "benchmark manifest threads must be positive"
}

$ocelotlCommand = Get-CommandArray -Target $manifest.ocelotl -Label "ocelotl"
$whisperCppCommand = Get-CommandArray -Target $manifest.whisper_cpp -Label "whisper.cpp"
$threads = [int]$manifest.threads

$plannedOcelotl = New-RunRecord `
    -Command $ocelotlCommand `
    -ModelPath ([string]$manifest.ocelotl.model_path) `
    -AudioPath ([string]$manifest.ocelotl.audio_path) `
    -Threads $threads `
    -Status "planned" `
    -WallTimeMs $null `
    -ExitCode $null `
    -Stdout $null `
    -SkipReason $null

$plannedWhisperCpp = New-RunRecord `
    -Command $whisperCppCommand `
    -ModelPath ([string]$manifest.whisper_cpp.model_path) `
    -AudioPath ([string]$manifest.whisper_cpp.audio_path) `
    -Threads $threads `
    -Status "planned" `
    -WallTimeMs $null `
    -ExitCode $null `
    -Stdout $null `
    -SkipReason $null

if ($DryRun) {
    $record = New-Record -Manifest $manifest -Status "planned" -Ocelotl $plannedOcelotl -WhisperCpp $plannedWhisperCpp
    Write-Record -Record $record -OutputPath $OutputPath
    exit 0
}

$whisperCppBinaryPath = Resolve-RepoPath -Root $repoRoot -Path ([string]$manifest.whisper_cpp.binary)
if (-not (Test-Path -LiteralPath $whisperCppBinaryPath -PathType Leaf)) {
    $skipReason = "missing whisper.cpp binary at $($manifest.whisper_cpp.binary); build whisper.cpp, copy whisper-cli.exe there, or edit the manifest binary path; see docs/benchmarks/whisper-cpp.md"
    $skippedWhisperCpp = New-RunRecord `
        -Command $whisperCppCommand `
        -ModelPath ([string]$manifest.whisper_cpp.model_path) `
        -AudioPath ([string]$manifest.whisper_cpp.audio_path) `
        -Threads $threads `
        -Status "skipped" `
        -WallTimeMs $null `
        -ExitCode $null `
        -Stdout $null `
        -SkipReason $skipReason
    $record = New-Record -Manifest $manifest -Status "skipped" -Ocelotl $plannedOcelotl -WhisperCpp $skippedWhisperCpp
    Write-Record -Record $record -OutputPath $OutputPath
    exit 0
}

$requiredInputs = @(
    [string]$manifest.ocelotl.model_path,
    [string]$manifest.ocelotl.audio_path,
    [string]$manifest.whisper_cpp.model_path
)
foreach ($inputPath in $requiredInputs) {
    $resolved = Resolve-RepoPath -Root $repoRoot -Path $inputPath
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        $skipReason = "missing benchmark input at $inputPath; prepare local artifacts per docs/benchmarks/whisper-cpp.md and docs/artifact-preparation.md"
        $skippedOcelotl = New-RunRecord `
            -Command $ocelotlCommand `
            -ModelPath ([string]$manifest.ocelotl.model_path) `
            -AudioPath ([string]$manifest.ocelotl.audio_path) `
            -Threads $threads `
            -Status "skipped" `
            -WallTimeMs $null `
            -ExitCode $null `
            -Stdout $null `
            -SkipReason $skipReason
        $skippedWhisperCpp = New-RunRecord `
            -Command $whisperCppCommand `
            -ModelPath ([string]$manifest.whisper_cpp.model_path) `
            -AudioPath ([string]$manifest.whisper_cpp.audio_path) `
            -Threads $threads `
            -Status "skipped" `
            -WallTimeMs $null `
            -ExitCode $null `
            -Stdout $null `
            -SkipReason $skipReason
        $record = New-Record -Manifest $manifest -Status "skipped" -Ocelotl $skippedOcelotl -WhisperCpp $skippedWhisperCpp
        Write-Record -Record $record -OutputPath $OutputPath
        exit 0
    }
}

Push-Location -LiteralPath $repoRoot
try {
    $ocelotlRun = Invoke-BenchmarkCommand `
        -Command $ocelotlCommand `
        -ModelPath ([string]$manifest.ocelotl.model_path) `
        -AudioPath ([string]$manifest.ocelotl.audio_path) `
        -Threads $threads

    if ($ocelotlRun.status -ne "completed") {
        $record = New-Record -Manifest $manifest -Status "failed" -Ocelotl $ocelotlRun -WhisperCpp $plannedWhisperCpp
        Write-Record -Record $record -OutputPath $OutputPath
        exit 1
    }

    $whisperCppRun = Invoke-BenchmarkCommand `
        -Command $whisperCppCommand `
        -ModelPath ([string]$manifest.whisper_cpp.model_path) `
        -AudioPath ([string]$manifest.whisper_cpp.audio_path) `
        -Threads $threads

    $status = if ($whisperCppRun.status -eq "completed") { "completed" } else { "failed" }
    $record = New-Record -Manifest $manifest -Status $status -Ocelotl $ocelotlRun -WhisperCpp $whisperCppRun
    Write-Record -Record $record -OutputPath $OutputPath
    if ($status -eq "completed") {
        exit 0
    }
    exit 1
} finally {
    Pop-Location
}
