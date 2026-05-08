# Ocelotl Agent Instructions

These instructions layer on top of the global Codex working agreements. They
are repo-specific: prefer this file when working inside `D:\Dev\ai\ocelotl`.

## Project Shape

Ocelotl is a Rust-first LLM inference runtime. It is correctness-first and
test-driven. CPU/reference behavior comes before GPU, contiguous KV comes before
paged KV, one request comes before scheduling, and narrow public APIs are
preferred until a milestone proves the next surface is needed.

Start from the local repo, not prior chat summaries:

1. Check `git status --short` and recent history.
2. Read `docs/start-here.md`.
3. Read the current milestone spec in `docs/milestones/`.
4. Read the matching task file in `docs/tasks/`.
5. Read the relevant design doc in `docs/design/`.
6. Use `docs/validation/test-matrix.md` as the acceptance traceability source.

Task checkboxes can drift. Milestone closure is proven by the validation matrix,
parity docs, and current test results.

As of 2026-05-07, M3 is closed and M4 has not started in this repo. Verify that
against the current tree before acting.

## Team Workspace

The durable team workspace lives in Obsidian at:

`projects/ocelotl/devs/`

The repo-local `.local/` workspace was retired on 2026-05-05. Do not recreate it.
Use Obsidian MCP tools as the primary path. If MCP is unavailable, the filesystem
fallback is:

`D:\Dev\knowledge_base\projects\ocelotl\devs\`

Before writing Obsidian notes, read `MASTERPLAN.md`. Follow the vault schema:
kebab-case filenames, required frontmatter, ISO dates, and no invented note
types or tags.

Important Obsidian notes for this repo:

- `projects/ocelotl/devs/assignments.md`
- `projects/ocelotl/devs/workflow/the-loop.md`
- `projects/ocelotl/devs/workflow/pairing.md`
- `projects/ocelotl/devs/workflow/promotion.md`
- `projects/ocelotl/devs/concepts/`
- `projects/ocelotl/devs/rust/`

Repo docs win if they conflict with the team workspace. Update the team
workspace after the repo is made true.

## Senior And Junior Workflow

The normal team pattern is: the main thread acts as senior dev, and explicitly
dispatched subagents act as juniors.

When junior agents are used:

- Create one git worktree per agent under `D:\Dev\ai\ocelotl-worktrees\`.
- Use branches named by milestone and owner, for example
  `m4/dev-03-rick-gpu-kernel` or `m4/pair-matt-james-dispatch`.
- Give each agent a narrow file ownership boundary.
- Tell agents they are not alone in the codebase and must not revert others'
  edits.
- Agents read Obsidian workspace docs, but the senior owns
  `projects/ocelotl/devs/assignments.md`.
- Agents report findings, tests, commits, and any smells. The senior audits,
  runs the gates, merges, and updates the assignment board.

Pair tasks are for integration seams and unobserved-assumption risk, not just
difficulty. Pairing paid off by catching dependency-direction mistakes and the
Qwen2.5 `model_type = "qwen2"` trap.

## Worktree Safety

Worktree path mistakes have happened. Before editing or committing in a
worktree, verify location with:

```powershell
git status --short
git branch --show-current
```

After resolving merges, search for conflict markers before committing:

```powershell
rg -n "(<){7}|(=){7}|(>){7}" .
```

Do not amend or hide already-created commits just to make history prettier.
Prefer honest follow-up commits for senior fixes after a merge.

## TDD Loop

Follow `docs/validation/tdd.md` and the Obsidian loop:

1. Write or update the spec for the behavior.
2. Add the smallest failing test.
3. Confirm it fails for the expected reason.
4. Implement the smallest correct change.
5. Re-run the focused test.
6. Run the relevant crate tests.
7. Run workspace validation.
8. Update docs if the contract changed.

A compile error can be the right first failing test. Do not skip the failure
check; it is how this project avoids building to a typo or a false assumption.

## Validation Gates

Use focused tests while developing. Before merge or commit handoff, run the
broad gate unless the change is docs-only and clearly does not need it:

```powershell
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
pwsh -NoProfile -File ci/check-offline.ps1
```

Run `cargo audit` when dependency changes or security posture is being checked.
If it reports an advisory, state whether it is a vulnerability, an allowed
unmaintained dependency warning, or a blocker.

Default tests must stay offline. Network-dependent tests must be `#[ignore]`,
documented, and gated on explicit local artifacts.

## Crate Boundaries

Respect `docs/crate-boundaries.md`.

- `ocelotl-core`: shared types, errors, device and metadata contracts.
- `ocelotl-loader`: local artifact discovery, parsing, and validation.
- `ocelotl-tokenizer`: tokenization and chat-template boundaries.
- `ocelotl-kernels`: low-level portable compute boundaries.
- `ocelotl-models`: model-family semantics and forward paths.
- `ocelotl-runtime`: request lifecycle, prefill/decode flow, KV, sampling.
- `ocelotl-server`: transport/API adaptation.

External crates stay behind Ocelotl-owned types. Do not re-export foreign crate
types across public boundaries unless a design doc explicitly chooses that.

Conversions should preserve dependency direction. If a bridge crosses crates,
put it in the outer crate that already depends inward.

## Model Families

Model-family code belongs under family modules, for example:

`crates/models/src/qwen/`

Future families such as Gemma should get their own modules rather than being
folded into Qwen-specific code. Preserve stable root exports when moving modules
so existing users can keep importing from `ocelotl_models::*`.

Qwen2.5-specific reminders:

- HF `model_type` is `"qwen2"`, not `"qwen2.5"`.
- Real Qwen2.5-0.5B-Instruct metadata uses BF16. The CPU reference path may
  upcast BF16 to F32 for compute, but BF16 is not automatically unsupported.
- Shape, dtype, context, RoPE, and token-id errors should fail before compute.

Gemma4/GGUF is a separate loader/model-family track, not a small Qwen2.5
extension. GGUF introduces embedded tokenizer metadata, quantized tensor types,
sliding-window/shared-KV details, and Gemma-specific softcapping.

## Loader And Artifact Policy

Inspection should be bounded and header-first. Do not read large tensor payloads
just to validate metadata. Validate declared lengths, offsets, shapes, dtypes,
and file size relationships before exposing a manifest.

Local model files belong under `local-artifacts/` or another documented
user-provided path. Do not commit license-bearing weights, tokenizer artifacts,
or downloaded model files unless a fixture policy explicitly allows it.

Safetensors is the first supported real artifact path. GGUF support should start
with a header-only inspector and a manifest contract before execution work.

## Kernel And Runtime Policy

The CPU path is the reference authority until a later milestone proves parity.
New GPU, cache, batching, or quantized paths must compare against CPU/reference
fixtures with documented tolerance.

Use checked shape arithmetic for slice lengths, matrix dimensions, context
lengths, and buffer sizes. Reject overflow and unsupported layouts before
compute.

Kernel tests should include hand-checked small tensors and bug-amplifying
tripwire data. Tolerances are math contracts, not flakiness knobs. If a
tolerance changes, document the computation chain and update
`docs/validation/parity.md` when it affects parity policy.

Runtime decode must go through the same public runtime/model path as prefill
unless a design doc explicitly introduces a separate cached path.

## Commits

Use conventional commits and keep them atomic. Use pathspecs with `git add`.
Read your own diff before committing.

Use `git commit -F-` with a single-quoted heredoc for messages containing
backticks, non-ASCII text, or complex body formatting. This avoids shell
interpolation and mojibake in commit history.

Do not force-add ignored artifacts. Do not rewrite user changes. Do not use
destructive git commands unless the user explicitly asks and the target has been
verified.
