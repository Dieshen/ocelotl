[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ManifestPath,

    [string]$OutputPath,

    [Nullable[int]]$WarmupIterations,

    [Nullable[int]]$MeasuredIterations,

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
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path -Path $Root -ChildPath $Path))
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
    $command = @($items | ForEach-Object { [string]$_ })
    if ($command | Where-Object { [string]::IsNullOrWhiteSpace($_) }) {
        throw "$Label command items must be non-empty"
    }
    return $command
}

function Get-RequiredInputArray {
    param([Parameter(Mandatory = $true)][object]$Target)

    if ($null -eq $Target.PSObject.Properties["required_inputs"]) {
        return @()
    }
    return @($Target.required_inputs | ForEach-Object { [string]$_ })
}

function Get-CommandFlagValue {
    param(
        [Parameter(Mandatory = $true)][object[]]$Command,
        [Parameter(Mandatory = $true)][string]$Flag,
        [Parameter(Mandatory = $true)][string]$Label
    )

    $positions = for ($index = 0; $index -lt $Command.Count; $index++) {
        if ([string]$Command[$index] -eq $Flag) {
            $index
        }
    }
    if (@($positions).Count -ne 1) {
        throw "$Label command must contain exactly one $Flag argument"
    }
    $position = [int]@($positions)[0]
    if ($position + 1 -ge $Command.Count) {
        throw "$Label command $Flag argument must have a value"
    }
    return [string]$Command[$position + 1]
}

function Assert-ManifestContract {
    param(
        [Parameter(Mandatory = $true)][object]$Manifest,
        [Parameter(Mandatory = $true)][object[]]$OcelotlCommand,
        [Parameter(Mandatory = $true)][object[]]$WhisperCppCommand
    )

    if ([int]$Manifest.fixture_version -ne 2) {
        throw "benchmark manifest fixture_version must be 2"
    }
    if ([int]$Manifest.threads -lt 1) {
        throw "benchmark manifest threads must be positive"
    }
    if ([int]$Manifest.warmup_iterations -lt 0) {
        throw "benchmark manifest warmup_iterations must be non-negative"
    }
    if ([int]$Manifest.measured_iterations -lt 1) {
        throw "benchmark manifest measured_iterations must be positive"
    }
    if ([string]$Manifest.execution_order -ne "alternating") {
        throw "benchmark manifest execution_order must be alternating"
    }
    if ([string]$Manifest.output_equivalence -ne "normalized_transcript") {
        throw "benchmark manifest output_equivalence must be normalized_transcript"
    }
    foreach ($field in @("name", "model_path", "audio_path", "model_family", "language", "decoding_mode")) {
        if ([string]::IsNullOrWhiteSpace([string]$Manifest.$field)) {
            throw "benchmark manifest $field must be non-empty"
        }
    }
    if ([string]$Manifest.ocelotl.audio_path -ne [string]$Manifest.audio_path -or
        [string]$Manifest.whisper_cpp.audio_path -ne [string]$Manifest.audio_path) {
        throw "both benchmark commands must use the manifest audio_path"
    }
    if ([string]$Manifest.ocelotl.model_path -ne [string]$Manifest.model_path) {
        throw "Ocelotl model_path must equal the manifest model_path"
    }
    if ([string]$Manifest.whisper_cpp.comparison_mode -ne [string]$Manifest.decoding_mode) {
        throw "whisper.cpp comparison_mode must equal manifest decoding_mode"
    }

    $ocelotlThreads = Get-CommandFlagValue -Command $OcelotlCommand -Flag "--cpu-threads" -Label "Ocelotl"
    if ($ocelotlThreads -ne ([int]$Manifest.threads).ToString()) {
        throw "Ocelotl command --cpu-threads must equal manifest threads"
    }
    $ocelotlBackend = Get-CommandFlagValue -Command $OcelotlCommand -Flag "--backend" -Label "Ocelotl"
    if ($ocelotlBackend -ne [string]$Manifest.ocelotl.backend) {
        throw "Ocelotl command --backend must equal ocelotl.backend"
    }
    $kernelMode = Get-CommandFlagValue -Command $OcelotlCommand -Flag "--cpu-kernel-mode" -Label "Ocelotl"
    if ($kernelMode -ne [string]$Manifest.ocelotl.cpu_kernel_mode) {
        throw "Ocelotl command --cpu-kernel-mode must equal ocelotl.cpu_kernel_mode"
    }

    $whisperThreads = Get-CommandFlagValue -Command $WhisperCppCommand -Flag "-t" -Label "whisper.cpp"
    if ($whisperThreads -ne ([int]$Manifest.threads).ToString()) {
        throw "whisper.cpp command -t must equal manifest threads"
    }
    if ([string]$Manifest.whisper_cpp.backend -eq "cpu" -and
        -not ($WhisperCppCommand -contains "-ng")) {
        throw "whisper.cpp CPU command must include -ng to disable GPU offload"
    }
    if ([string]$Manifest.whisper_cpp.backend -ne "cpu") {
        throw "benchmark manifest currently supports only whisper.cpp backend cpu"
    }
    if (-not ($WhisperCppCommand -contains "-otxt")) {
        throw "whisper.cpp command must include -otxt so output equivalence can be checked"
    }
    if ([string]::IsNullOrWhiteSpace([string]$Manifest.whisper_cpp.transcript_path)) {
        throw "whisper.cpp transcript_path must be non-empty"
    }
}

function New-EmptyOutputSummary {
    return [ordered]@{
        token_count = $null
        text = $null
        stdout_excerpt = $null
        parsed_json = $null
    }
}

function New-OutputSummary {
    param(
        [AllowNull()][string]$Stdout,
        [Parameter(Mandatory = $true)][string]$Engine,
        [AllowNull()][string]$TranscriptPath,
        [Parameter(Mandatory = $true)][datetime]$StartedAtUtc
    )

    $excerpt = $null
    $parsedJson = $null
    $tokenCount = $null
    $text = $null
    if (-not [string]::IsNullOrWhiteSpace($Stdout)) {
        $raw = $Stdout.Trim()
        $excerpt = $raw
        if ($excerpt.Length -gt 2000) {
            $excerpt = $excerpt.Substring(0, 2000)
        }
        if ($Engine -eq "ocelotl") {
            try {
                $parsedJson = $raw | ConvertFrom-Json -ErrorAction Stop
                if ($null -ne $parsedJson.PSObject.Properties["token_count"]) {
                    $tokenCount = [Nullable[int64]]$parsedJson.token_count
                }
                if ($null -ne $parsedJson.PSObject.Properties["text"] -and $null -ne $parsedJson.text) {
                    $text = [string]$parsedJson.text
                }
            } catch {
                $parsedJson = $null
            }
        }
    }

    if ($Engine -eq "whisper_cpp" -and -not [string]::IsNullOrWhiteSpace($TranscriptPath)) {
        if (Test-Path -LiteralPath $TranscriptPath -PathType Leaf) {
            $transcript = Get-Item -LiteralPath $TranscriptPath
            if ($transcript.LastWriteTimeUtc -ge $StartedAtUtc) {
                $text = (Get-Content -Raw -LiteralPath $TranscriptPath).Trim()
            }
        }
    }

    return [ordered]@{
        token_count = $tokenCount
        text = $text
        stdout_excerpt = $excerpt
        parsed_json = $parsedJson
    }
}

function Get-OcelotlOutputValidationError {
    param(
        [Parameter(Mandatory = $true)][object]$Output,
        [Parameter(Mandatory = $true)][object]$Manifest
    )

    $parsed = $Output.parsed_json
    if ($null -eq $parsed) {
        return "Ocelotl benchmark output must be one JSON object"
    }
    foreach ($field in @("backend", "cpu_kernel_mode", "cpu_threads", "matches_expected", "timings_ms")) {
        if ($null -eq $parsed.PSObject.Properties[$field]) {
            return "Ocelotl benchmark output is missing $field"
        }
    }
    if ([string]$parsed.backend -ne [string]$Manifest.ocelotl.backend) {
        return "Ocelotl output backend does not match the manifest"
    }
    if ([string]$parsed.cpu_kernel_mode -ne [string]$Manifest.ocelotl.cpu_kernel_mode) {
        return "Ocelotl output cpu_kernel_mode does not match the manifest"
    }
    if ([int]$parsed.cpu_threads -ne [int]$Manifest.threads) {
        return "Ocelotl output cpu_threads does not match the manifest"
    }
    if (-not [bool]$parsed.matches_expected) {
        return "Ocelotl output failed exact expected-token parity"
    }
    return $null
}

function Invoke-BenchmarkSample {
    param(
        [Parameter(Mandatory = $true)][object[]]$Command,
        [Parameter(Mandatory = $true)][string]$Engine,
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][int]$Iteration,
        [Parameter(Mandatory = $true)][int]$Sequence,
        [AllowNull()][string]$TranscriptPath
    )

    $executable = [string]$Command[0]
    $arguments = if ($Command.Count -gt 1) { @($Command[1..($Command.Count - 1)]) } else { @() }
    $startedAtUtc = [datetime]::UtcNow
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $stdout = & $executable @arguments 2>&1 | Out-String
        $exitCode = $LASTEXITCODE
        if ($null -eq $exitCode) {
            $exitCode = 0
        }
        $stopwatch.Stop()
        return [ordered]@{
            sequence = $Sequence
            phase = $Phase
            iteration = $Iteration
            engine = $Engine
            status = if ($exitCode -eq 0) { "completed" } else { "failed" }
            wall_time_ms = [int64]$stopwatch.ElapsedMilliseconds
            exit_code = [int]$exitCode
            output = New-OutputSummary -Stdout $stdout -Engine $Engine -TranscriptPath $TranscriptPath -StartedAtUtc $startedAtUtc
            validation_error = $null
        }
    } catch {
        $stopwatch.Stop()
        return [ordered]@{
            sequence = $Sequence
            phase = $Phase
            iteration = $Iteration
            engine = $Engine
            status = "failed"
            wall_time_ms = [int64]$stopwatch.ElapsedMilliseconds
            exit_code = 1
            output = New-OutputSummary -Stdout $_.Exception.Message -Engine $Engine -TranscriptPath $null -StartedAtUtc $startedAtUtc
            validation_error = $_.Exception.Message
        }
    }
}

function Get-Percentile {
    param(
        [Parameter(Mandatory = $true)][long[]]$SortedValues,
        [Parameter(Mandatory = $true)][double]$Percentile
    )

    $rank = [Math]::Ceiling($Percentile * $SortedValues.Count)
    $index = [Math]::Max(0, [Math]::Min($SortedValues.Count - 1, $rank - 1))
    return [double]$SortedValues[$index]
}

function New-Aggregate {
    param(
        [AllowEmptyCollection()][object[]]$Samples,
        [Parameter(Mandatory = $true)][string]$Engine
    )

    $values = @($Samples |
        Where-Object { $_.phase -eq "measured" -and $_.engine -eq $Engine -and $_.status -eq "completed" } |
        ForEach-Object { [long]$_.wall_time_ms } |
        Sort-Object)
    if ($values.Count -eq 0) {
        return $null
    }
    $middle = [Math]::Floor($values.Count / 2)
    $median = if ($values.Count % 2 -eq 1) {
        [double]$values[$middle]
    } else {
        ([double]$values[$middle - 1] + [double]$values[$middle]) / 2.0
    }
    $sum = ($values | Measure-Object -Sum).Sum
    return [ordered]@{
        sample_count = $values.Count
        min_wall_time_ms = [long]$values[0]
        mean_wall_time_ms = [double]$sum / $values.Count
        median_wall_time_ms = $median
        p95_wall_time_ms = Get-Percentile -SortedValues $values -Percentile 0.95
        max_wall_time_ms = [long]$values[$values.Count - 1]
    }
}

function New-EnvironmentRecord {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$ManifestFullPath
    )

    $revision = ((& git -C $RepoRoot rev-parse HEAD 2>$null) | Out-String).Trim()
    if ([string]::IsNullOrWhiteSpace($revision)) {
        $revision = "unknown"
    }
    $dirtyOutput = ((& git -C $RepoRoot status --porcelain 2>$null) | Out-String).Trim()
    return [ordered]@{
        captured_at_utc = [datetime]::UtcNow.ToString("o")
        repository_revision = $revision
        repository_dirty = -not [string]::IsNullOrWhiteSpace($dirtyOutput)
        os = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
        architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
        powershell_version = $PSVersionTable.PSVersion.ToString()
        logical_processors = [Environment]::ProcessorCount
        processor = [string]$env:PROCESSOR_IDENTIFIER
        repo_root = $RepoRoot
        manifest_path = $ManifestFullPath
    }
}

function New-EngineRecord {
    param(
        [Parameter(Mandatory = $true)][object]$Manifest,
        [Parameter(Mandatory = $true)][string]$Engine,
        [Parameter(Mandatory = $true)][object[]]$Command,
        [AllowEmptyCollection()][object[]]$Samples,
        [Parameter(Mandatory = $true)][string]$Status,
        [AllowNull()][object]$SkipReason,
        [Parameter(Mandatory = $true)][string]$RepoRoot
    )

    $target = if ($Engine -eq "ocelotl") { $Manifest.ocelotl } else { $Manifest.whisper_cpp }
    $engineSamples = @($Samples | Where-Object { $_.engine -eq $Engine })
    $latestOutput = if ($engineSamples.Count -gt 0) {
        $engineSamples[$engineSamples.Count - 1].output
    } else {
        New-EmptyOutputSummary
    }
    $modelFullPath = Resolve-RepoPath -Root $RepoRoot -Path ([string]$target.model_path)
    $modelSha256 = if ($Status -notin @("planned", "skipped") -and
        (Test-Path -LiteralPath $modelFullPath -PathType Leaf)) {
        (Get-FileHash -Algorithm SHA256 -LiteralPath $modelFullPath).Hash.ToLowerInvariant()
    } else {
        $null
    }
    return [ordered]@{
        command = @($Command)
        model_path = [string]$target.model_path
        model_sha256 = $modelSha256
        audio_path = [string]$target.audio_path
        effective_config = [ordered]@{
            backend = [string]$target.backend
            threads = [int]$Manifest.threads
            cpu_kernel_mode = if ($Engine -eq "ocelotl") { [string]$target.cpu_kernel_mode } else { $null }
            comparison_mode = if ($Engine -eq "whisper_cpp") { [string]$target.comparison_mode } else { $null }
        }
        status = $Status
        aggregate = New-Aggregate -Samples $Samples -Engine $Engine
        latest_output = $latestOutput
        skip_reason = $SkipReason
    }
}

function Normalize-Transcript {
    param([AllowNull()][string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $null
    }
    $normalized = $Text.Normalize([Text.NormalizationForm]::FormKC).ToLowerInvariant()
    return (($normalized -replace '[^\p{L}\p{Nd}]+', ' ' -replace '\s+', ' ').Trim())
}

function New-ComparisonRecord {
    param(
        [Parameter(Mandatory = $true)][object]$Manifest,
        [AllowEmptyCollection()][object[]]$Samples,
        [Parameter(Mandatory = $true)][string]$Status,
        [AllowNull()][object]$Reason,
        [AllowNull()][object]$AudioSha256
    )

    $outputMatch = $null
    if ($Status -eq "completed") {
        $ocelotlSamples = @($Samples |
            Where-Object { $_.phase -eq "measured" -and $_.engine -eq "ocelotl" })
        $whisperSamples = @($Samples |
            Where-Object { $_.phase -eq "measured" -and $_.engine -eq "whisper_cpp" })
        $ocelotlTexts = @($ocelotlSamples |
            ForEach-Object { Normalize-Transcript -Text $_.output.text } |
            Where-Object { $null -ne $_ } |
            Select-Object -Unique)
        $whisperTexts = @($whisperSamples |
            ForEach-Object { Normalize-Transcript -Text $_.output.text } |
            Where-Object { $null -ne $_ } |
            Select-Object -Unique)
        $ocelotlTextCount = @($ocelotlSamples | Where-Object { $null -ne (Normalize-Transcript -Text $_.output.text) }).Count
        $whisperTextCount = @($whisperSamples | Where-Object { $null -ne (Normalize-Transcript -Text $_.output.text) }).Count
        if ($ocelotlSamples.Count -gt 0 -and
            $ocelotlTextCount -eq $ocelotlSamples.Count -and
            $whisperTextCount -eq $whisperSamples.Count -and
            $ocelotlTexts.Count -eq 1 -and
            $whisperTexts.Count -eq 1) {
            $outputMatch = $ocelotlTexts[0] -eq $whisperTexts[0]
            if (-not $outputMatch) {
                $Reason = "normalized Ocelotl and whisper.cpp transcripts differ"
            }
        } else {
            $outputMatch = $false
            $Reason = "each engine must produce one stable non-empty transcript across measured samples"
        }
    }
    $comparisonStatus = switch ($Status) {
        "completed" { if ($outputMatch) { "comparable" } else { "incomparable" } }
        "planned" { "planned" }
        "skipped" { "skipped" }
        default { "failed" }
    }
    return [ordered]@{
        status = $comparisonStatus
        workload_match = $true
        model_family = [string]$Manifest.model_family
        language = [string]$Manifest.language
        decoding_mode = [string]$Manifest.decoding_mode
        output_equivalence = [string]$Manifest.output_equivalence
        output_match = $outputMatch
        audio_sha256 = $AudioSha256
        reason = $Reason
    }
}

function New-Record {
    param(
        [Parameter(Mandatory = $true)][object]$Manifest,
        [Parameter(Mandatory = $true)][string]$Status,
        [AllowEmptyCollection()][object[]]$Samples,
        [Parameter(Mandatory = $true)][object]$Environment,
        [Parameter(Mandatory = $true)][object[]]$OcelotlCommand,
        [Parameter(Mandatory = $true)][object[]]$WhisperCppCommand,
        [AllowNull()][object]$OcelotlSkipReason,
        [AllowNull()][object]$WhisperCppSkipReason,
        [AllowNull()][object]$ComparisonReason,
        [AllowNull()][object]$AudioSha256,
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][int]$Warmups,
        [Parameter(Mandatory = $true)][int]$Measurements
    )

    $ocelotlStatus = if ($null -ne $OcelotlSkipReason) { "skipped" } elseif ($Status -eq "planned") { "planned" } elseif (@($Samples | Where-Object { $_.engine -eq "ocelotl" -and $_.status -eq "failed" }).Count -gt 0) { "failed" } else { $Status }
    $whisperStatus = if ($null -ne $WhisperCppSkipReason) { "skipped" } elseif ($Status -eq "planned") { "planned" } elseif (@($Samples | Where-Object { $_.engine -eq "whisper_cpp" -and $_.status -eq "failed" }).Count -gt 0) { "failed" } else { $Status }
    return [ordered]@{
        fixture_version = 2
        manifest_name = [string]$Manifest.name
        status = $Status
        benchmark_plan = [ordered]@{
            warmup_iterations = $Warmups
            measured_iterations = $Measurements
            execution_order = "alternating"
        }
        environment = $Environment
        comparison = New-ComparisonRecord -Manifest $Manifest -Samples $Samples -Status $Status -Reason $ComparisonReason -AudioSha256 $AudioSha256
        samples = @($Samples)
        ocelotl = New-EngineRecord -Manifest $Manifest -Engine "ocelotl" -Command $OcelotlCommand -Samples $Samples -Status $ocelotlStatus -SkipReason $OcelotlSkipReason -RepoRoot $RepoRoot
        whisper_cpp = New-EngineRecord -Manifest $Manifest -Engine "whisper_cpp" -Command $WhisperCppCommand -Samples $Samples -Status $whisperStatus -SkipReason $WhisperCppSkipReason -RepoRoot $RepoRoot
    }
}

function Write-Record {
    param(
        [Parameter(Mandatory = $true)][object]$Record,
        [AllowNull()][string]$OutputPath
    )

    $json = $Record | ConvertTo-Json -Depth 32
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
$ocelotlCommand = Get-CommandArray -Target $manifest.ocelotl -Label "Ocelotl"
$whisperCppCommand = Get-CommandArray -Target $manifest.whisper_cpp -Label "whisper.cpp"
Assert-ManifestContract -Manifest $manifest -OcelotlCommand $ocelotlCommand -WhisperCppCommand $whisperCppCommand

$warmups = if ($null -ne $WarmupIterations) { [int]$WarmupIterations } else { [int]$manifest.warmup_iterations }
$measurements = if ($null -ne $MeasuredIterations) { [int]$MeasuredIterations } else { [int]$manifest.measured_iterations }
if ($warmups -lt 0) {
    throw "WarmupIterations must be non-negative"
}
if ($measurements -lt 1) {
    throw "MeasuredIterations must be positive"
}

$environment = New-EnvironmentRecord -RepoRoot $repoRoot -ManifestFullPath $manifestFullPath
$samples = @()

if ($DryRun) {
    $record = New-Record -Manifest $manifest -Status "planned" -Samples $samples -Environment $environment -OcelotlCommand $ocelotlCommand -WhisperCppCommand $whisperCppCommand -OcelotlSkipReason $null -WhisperCppSkipReason $null -ComparisonReason $null -AudioSha256 $null -RepoRoot $repoRoot -Warmups $warmups -Measurements $measurements
    Write-Record -Record $record -OutputPath $OutputPath
    exit 0
}

$whisperCppBinaryPath = Resolve-RepoPath -Root $repoRoot -Path ([string]$manifest.whisper_cpp.binary)
$ocelotlBinaryPath = Resolve-RepoPath -Root $repoRoot -Path ([string]$ocelotlCommand[0])
$skipReason = $null
$skipEngine = $null
if (-not (Test-Path -LiteralPath $whisperCppBinaryPath -PathType Leaf)) {
    $skipEngine = "whisper_cpp"
    $skipReason = "missing whisper.cpp binary at $($manifest.whisper_cpp.binary); build whisper.cpp, copy whisper-cli.exe there, or edit the manifest binary path; see docs/benchmarks/whisper-cpp.md"
} elseif (-not (Test-Path -LiteralPath $ocelotlBinaryPath -PathType Leaf)) {
    $skipEngine = "ocelotl"
    $skipReason = "missing Ocelotl benchmark hook at $($ocelotlCommand[0]); build it with cargo build --release, then rerun this benchmark; see docs/benchmarks/whisper-cpp.md"
} else {
    $requiredInputs = @(
        [string]$manifest.ocelotl.model_path,
        [string]$manifest.ocelotl.audio_path,
        [string]$manifest.whisper_cpp.model_path
    ) + (Get-RequiredInputArray -Target $manifest.ocelotl)
    foreach ($inputPath in ($requiredInputs | Select-Object -Unique)) {
        $resolved = Resolve-RepoPath -Root $repoRoot -Path $inputPath
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
            $skipEngine = "both"
            $skipReason = "missing benchmark input at $inputPath; prepare local artifacts per docs/benchmarks/whisper-cpp.md and docs/artifact-preparation.md"
            break
        }
    }
}

if ($null -ne $skipReason) {
    $ocelotlSkip = if ($skipEngine -eq "whisper_cpp") { $null } else { $skipReason }
    $whisperSkip = if ($skipEngine -eq "ocelotl") { $null } else { $skipReason }
    $record = New-Record -Manifest $manifest -Status "skipped" -Samples $samples -Environment $environment -OcelotlCommand $ocelotlCommand -WhisperCppCommand $whisperCppCommand -OcelotlSkipReason $ocelotlSkip -WhisperCppSkipReason $whisperSkip -ComparisonReason $skipReason -AudioSha256 $null -RepoRoot $repoRoot -Warmups $warmups -Measurements $measurements
    Write-Record -Record $record -OutputPath $OutputPath
    exit 0
}

$audioFullPath = Resolve-RepoPath -Root $repoRoot -Path ([string]$manifest.audio_path)
$audioSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $audioFullPath).Hash.ToLowerInvariant()
$transcriptFullPath = Resolve-RepoPath -Root $repoRoot -Path ([string]$manifest.whisper_cpp.transcript_path)
$sequence = 0
$failureReason = $null

Push-Location -LiteralPath $repoRoot
try {
    foreach ($phasePlan in @(
        [ordered]@{ phase = "warmup"; count = $warmups },
        [ordered]@{ phase = "measured"; count = $measurements }
    )) {
        for ($iteration = 0; $iteration -lt [int]$phasePlan.count; $iteration++) {
            $engines = if ($iteration % 2 -eq 0) { @("ocelotl", "whisper_cpp") } else { @("whisper_cpp", "ocelotl") }
            foreach ($engine in $engines) {
                $command = if ($engine -eq "ocelotl") { $ocelotlCommand } else { $whisperCppCommand }
                $sample = Invoke-BenchmarkSample -Command $command -Engine $engine -Phase ([string]$phasePlan.phase) -Iteration $iteration -Sequence $sequence -TranscriptPath $(if ($engine -eq "whisper_cpp") { $transcriptFullPath } else { $null })
                if ($engine -eq "ocelotl" -and $sample.status -eq "completed") {
                    $validationError = Get-OcelotlOutputValidationError -Output $sample.output -Manifest $manifest
                    if ($null -ne $validationError) {
                        $sample.status = "failed"
                        $sample.validation_error = $validationError
                    }
                }
                $samples += $sample
                $sequence++
                if ($sample.status -ne "completed") {
                    $failureReason = if ($null -ne $sample.validation_error) { [string]$sample.validation_error } else { "$engine command failed" }
                    break
                }
            }
            if ($null -ne $failureReason) {
                break
            }
        }
        if ($null -ne $failureReason) {
            break
        }
    }
} finally {
    Pop-Location
}

if ($null -ne $failureReason) {
    $record = New-Record -Manifest $manifest -Status "failed" -Samples $samples -Environment $environment -OcelotlCommand $ocelotlCommand -WhisperCppCommand $whisperCppCommand -OcelotlSkipReason $null -WhisperCppSkipReason $null -ComparisonReason $failureReason -AudioSha256 $audioSha256 -RepoRoot $repoRoot -Warmups $warmups -Measurements $measurements
    Write-Record -Record $record -OutputPath $OutputPath
    exit 1
}

$record = New-Record -Manifest $manifest -Status "completed" -Samples $samples -Environment $environment -OcelotlCommand $ocelotlCommand -WhisperCppCommand $whisperCppCommand -OcelotlSkipReason $null -WhisperCppSkipReason $null -ComparisonReason $null -AudioSha256 $audioSha256 -RepoRoot $repoRoot -Warmups $warmups -Measurements $measurements
if ($record.comparison.status -ne "comparable") {
    $record.status = "incomparable"
    Write-Record -Record $record -OutputPath $OutputPath
    exit 2
}
Write-Record -Record $record -OutputPath $OutputPath
exit 0
