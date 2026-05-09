# Fixtures

Fixtures are small, deterministic test inputs used by Ocelotl's TDD workflow.
Default tests must run offline. Large model weights should not be committed.

## Layout

- `manifest/`: machine-readable artifact pins (repository + commit SHA +
  license) for external models referenced by other fixtures.
- `metadata/`: normalized model metadata and malformed metadata cases.
- `tokenizer/`: exact token ID and chat-template fixtures.
- `logits/`: reference logits or token outputs for small deterministic cases.
- `benchmarks/`: benchmark manifest and record schema examples. These validate
  harness shape only; real benchmark outputs stay under ignored local artifacts.

## First Target

The provisional first model target is documented in `docs/model-target.md`. M1
uses synthetic Qwen2.5-shaped fixtures. M2/M3 may add fixtures derived from an
exact pinned `Qwen/Qwen2.5-0.5B-Instruct` revision.
