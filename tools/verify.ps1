[CmdletBinding()]
param(
    [ValidateSet('Fast', 'Full')]
    [string]$Mode = 'Fast'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory)]
        [string]$Label,

        [Parameter(Mandatory)]
        [string]$Executable,

        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Executable @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode."
    }
}

Push-Location $RepoRoot
try {
    Invoke-CheckedCommand `
        -Label 'Format check' `
        -Executable 'cargo' `
        -Arguments @('fmt', '--all', '--', '--check')

    Invoke-CheckedCommand `
        -Label 'Default workspace check' `
        -Executable 'cargo' `
        -Arguments @('check', '--workspace', '--all-targets', '--locked')

    Invoke-CheckedCommand `
        -Label 'Offline-by-default policy' `
        -Executable 'pwsh' `
        -Arguments @('-NoProfile', '-File', (Join-Path $RepoRoot 'ci/check-offline.ps1'))

    if ($Mode -eq 'Full') {
        Invoke-CheckedCommand `
            -Label 'Default workspace tests' `
            -Executable 'cargo' `
            -Arguments @('test', '--workspace', '--locked')

        Invoke-CheckedCommand `
            -Label 'All-feature no-launch check' `
            -Executable 'cargo' `
            -Arguments @(
                'check',
                '--workspace',
                '--all-targets',
                '--all-features',
                '--locked'
            )

        Invoke-CheckedCommand `
            -Label 'Clippy (all targets and features)' `
            -Executable 'cargo' `
            -Arguments @(
                'clippy',
                '--workspace',
                '--all-targets',
                '--all-features',
                '--locked',
                '--',
                '-D',
                'warnings'
            )
    }

    Write-Host "Verification mode '$Mode' passed." -ForegroundColor Green
}
finally {
    Pop-Location
}
