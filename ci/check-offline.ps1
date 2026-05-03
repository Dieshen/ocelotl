# M2.8 — Offline-by-default CI gate.
#
# Scans the Rust workspace for patterns that would let a default
# `cargo test --workspace` run reach the network. Fails (exit 1) if any
# disallowed pattern is found in production source or in non-`#[ignore]`'d
# tests.
#
# Why this exists: per docs/ci.md the project's default test command must
# never fetch model artifacts from the network. The principle has been
# enforced by reviewer attention through M2 Phase 2A. This gate makes the
# rule mechanical so future contributions can't silently regress it.
#
# Layered defense:
#   - Layer 1 (CI command): the workflow runs `cargo test --workspace`
#     WITHOUT `-- --ignored`, so any test marked `#[ignore]` is skipped.
#   - Layer 2 (THIS script): scan the source for known network-fetching
#     APIs. If found, the call site MUST be inside an `#[ignore]`'d test
#     (or be a documentation/comment reference, which the script narrows
#     out by pattern shape).
#
# Limitations (honest):
#   - Greppable: a contributor who really wants to bypass it can. The goal
#     is catching accidents, not adversaries. If we need adversarial
#     guarantees, follow up with a sandbox-CI step (e.g. `--network none`
#     in a Docker job).
#   - Heuristic line-based: the script flags forbidden patterns at line
#     granularity, then checks whether the enclosing test is `#[ignore]`'d
#     by walking up to the nearest preceding `#[test]` / `#[ignore]` /
#     `#[cfg(test)]` attribute. Works for the patterns the team writes
#     today; would need refinement if we adopt macro-generated tests.
#
# How to add a legitimate exception: mark the test `#[ignore = "..."]` per
# docs/artifact-preparation.md § 5. The gate will permit it.

$ErrorActionPreference = 'Stop'

# Resolve repo root: this script lives in <repo>/ci/check-offline.ps1.
$RepoRoot = Split-Path -Parent $PSScriptRoot
$CratesDir = Join-Path $RepoRoot 'crates'

if (-not (Test-Path $CratesDir)) {
    Write-Error "crates/ not found at $CratesDir — running from wrong directory?"
    exit 2
}

# --- Forbidden patterns ---------------------------------------------------
#
# Each entry is (regex, human-readable description). Patterns target Rust
# crate names and identifiers, NOT human-language words, so doc comments
# that mention `huggingface-cli download ...` (a CLI command) don't false-
# positive against the `huggingface_hub` Python/Rust crate.

$ForbiddenPatterns = @(
    @{
        Pattern = '(?<![A-Za-z0-9_])reqwest::'
        Reason  = 'reqwest HTTP client'
    },
    @{
        Pattern = '(?<![A-Za-z0-9_])ureq::'
        Reason  = 'ureq HTTP client'
    },
    @{
        Pattern = '(?<![A-Za-z0-9_])isahc::'
        Reason  = 'isahc HTTP client'
    },
    @{
        Pattern = '(?<![A-Za-z0-9_])surf::'
        Reason  = 'surf HTTP client'
    },
    @{
        Pattern = '(?<![A-Za-z0-9_])attohttpc::'
        Reason  = 'attohttpc HTTP client'
    },
    @{
        Pattern = '(?<![A-Za-z0-9_])hyper::Client'
        Reason  = 'hyper HTTP client'
    },
    @{
        Pattern = '(?<![A-Za-z0-9_])hf_hub::'
        Reason  = 'hf-hub HuggingFace Hub client'
    },
    @{
        Pattern = '(?<![A-Za-z0-9_])HfApi(?![A-Za-z0-9_])'
        Reason  = 'huggingface_hub HfApi'
    },
    @{
        Pattern = '\.from_pretrained\b'
        Reason  = 'tokenizers::Tokenizer::from_pretrained (network fetch)'
    },
    @{
        Pattern = 'https?://huggingface\.co'
        Reason  = 'literal huggingface.co URL'
    },
    @{
        Pattern = 'https?://hf\.co'
        Reason  = 'literal hf.co URL'
    }
)

# --- Cargo.toml scan ------------------------------------------------------
#
# Disallow ANY of the above HTTP/HF-Hub crates from appearing as workspace
# deps. Tests can't accidentally use a crate that isn't in scope; blocking
# the dep is the cheapest enforcement.

$ForbiddenCargoDeps = @(
    'reqwest', 'ureq', 'isahc', 'surf', 'attohttpc',
    'hf-hub', 'hf_hub', 'huggingface_hub'
)

$Violations = New-Object System.Collections.Generic.List[string]

# Scan Cargo.toml files.
Get-ChildItem -Path $CratesDir -Recurse -Filter 'Cargo.toml' -File | ForEach-Object {
    $cargoPath = $_.FullName
    $relPath = $cargoPath.Substring($RepoRoot.Length).TrimStart('\', '/')
    $lines = Get-Content -LiteralPath $cargoPath
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        # Skip comments.
        if ($line -match '^\s*#') { continue }
        foreach ($dep in $ForbiddenCargoDeps) {
            # Match "<dep> =" at line start (allowing whitespace) — standard
            # Cargo.toml dep declaration shape.
            if ($line -match "^\s*$([regex]::Escape($dep))\s*=") {
                $Violations.Add(
                    "$relPath`:$($i + 1): forbidden dependency `'$dep`' " +
                    "(network-fetching crate; see ci/check-offline.ps1)"
                )
            }
        }
    }
}

# --- Source scan ----------------------------------------------------------
#
# For each *.rs under crates/, find lines matching any forbidden pattern.
# If the match is inside a test, verify the enclosing test is `#[ignore]`'d.
# If the match is in production (non-test) code, fail unconditionally.

function Test-IsIgnoredContext {
    param(
        [string[]]$AllLines,
        [int]$MatchLineIndex
    )

    # Walk backwards from the match line until we find either:
    #   - a `#[test]` (with or without `#[ignore]` adjacent above it), or
    #   - a `fn ` declaration at module level, or
    #   - the top of the file.
    # Returns $true if the nearest `#[test]` has an `#[ignore]` attribute
    # on an adjacent line above it, OR if no `#[test]` is found within
    # 200 lines (suggests the line is in a non-test region).

    for ($j = $MatchLineIndex; $j -ge 0 -and $j -ge ($MatchLineIndex - 200); $j--) {
        $l = $AllLines[$j].Trim()
        if ($l -match '^#\[test\]') {
            # Found the enclosing `#[test]`. The `#[ignore]` attribute may
            # appear EITHER on adjacent lines above `#[test]` OR on
            # adjacent attribute lines below `#[test]` (before the `fn`).
            # Both positions are idiomatic Rust; the canonical example in
            # crates/tokenizer/tests/qwen2_5_basic_prompt.rs places
            # `#[ignore]` directly below `#[test]`. Allow up to 5
            # intervening attribute lines on either side.

            # Check ABOVE.
            for ($k = $j - 1; $k -ge 0 -and $k -ge ($j - 5); $k--) {
                $above = $AllLines[$k].Trim()
                if ($above -match '^#\[ignore') { return $true }
                if ($above -eq '' -or $above -match '^#\[') { continue }
                break
            }

            # Check BELOW (between `#[test]` and `fn`).
            for ($k = $j + 1; $k -lt $AllLines.Count -and $k -le ($j + 5); $k++) {
                $below = $AllLines[$k].Trim()
                if ($below -match '^#\[ignore') { return $true }
                if ($below -match '^fn\s' -or $below -match '^\s*pub\s+fn\s') { break }
                if ($below -eq '' -or $below -match '^#\[') { continue }
                break
            }
            return $false
        }
    }
    # No `#[test]` found within window — treat as production code (must
    # fail). The forbidden patterns have no legitimate non-test use in
    # this project today.
    return $false
}

function Test-IsAllowedDocComment {
    param([string]$Line)
    # Doc-comment lines (`///` or `//!`) and inner-block comments that
    # merely describe an offline command (e.g. mentioning `huggingface-cli
    # download` in remediation text) are allowed. The forbidden patterns
    # above are crate-identifier shaped; the few legitimate documentation
    # mentions (the `huggingface-cli` CLI) don't match those patterns.
    # This helper exists for future tightening (e.g. if someone documents
    # a URL); currently it just whitelists doc-comment prefixes.
    return ($Line -match '^\s*///' -or $Line -match '^\s*//!')
}

Get-ChildItem -Path $CratesDir -Recurse -Filter '*.rs' -File | ForEach-Object {
    $rsPath = $_.FullName
    $relPath = $rsPath.Substring($RepoRoot.Length).TrimStart('\', '/')
    $lines = Get-Content -LiteralPath $rsPath

    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        foreach ($entry in $ForbiddenPatterns) {
            if ($line -match $entry.Pattern) {
                # Allow doc comments — they describe behavior, they don't
                # invoke it.
                if (Test-IsAllowedDocComment -Line $line) { continue }

                # If we're inside a test region, check that the enclosing
                # test is `#[ignore]`'d.
                if (Test-IsIgnoredContext -AllLines $lines -MatchLineIndex $i) {
                    continue
                }

                $Violations.Add(
                    "$relPath`:$($i + 1): forbidden pattern '$($entry.Reason)' " +
                    "outside an #[ignore]'d test — line: $($line.Trim())"
                )
            }
        }
    }
}

# --- Report ---------------------------------------------------------------

if ($Violations.Count -gt 0) {
    Write-Host ''
    Write-Host '=== Offline-gate violations ===' -ForegroundColor Red
    foreach ($v in $Violations) {
        Write-Host "  $v" -ForegroundColor Red
    }
    Write-Host ''
    Write-Host 'These lines would let `cargo test --workspace` reach the network.' -ForegroundColor Red
    Write-Host 'Mark the enclosing test `#[ignore = "..."]` per' -ForegroundColor Red
    Write-Host 'docs/artifact-preparation.md section 5, or remove the dependency.' -ForegroundColor Red
    Write-Host ''
    exit 1
}

Write-Host 'Offline gate: OK (no network-fetching patterns in default test surface).' -ForegroundColor Green
exit 0
