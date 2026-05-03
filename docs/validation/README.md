# Validation Docs

This directory holds the validation policy for Ocelotl. It documents *how* we
prove the runtime is correct, *what* tests are required at each milestone, and
*where* the boundaries are between correctness, hygiene, and benchmarking.

If you're new here, read in this order:

1. [`tdd.md`](tdd.md) — why this project mandates test-first.
2. [`principles.md`](principles.md) — cross-cutting validation principles
   (correctness vs. hygiene gates, offline-by-default).
3. [`test-matrix.md`](test-matrix.md) — required tests per milestone, plus
   the M1 acceptance traceability matrix.

## File Index

- **[`tdd.md`](tdd.md)** — TDD policy, the seven-step loop, required test
  categories per crate area, and the merge gate.
- **[`principles.md`](principles.md)** — correctness gates vs. hygiene gates,
  cross-references the offline-by-default policy in `docs/ci.md`.
- **[`test-matrix.md`](test-matrix.md)** — required tests per milestone (M0
  through M8), validation tiers (focused vs. workspace), the offline rule,
  and the M1 acceptance traceability table.
- **[`correctness.md`](correctness.md)** — the seven validation layers, the
  fixture policy, tolerance rules, and the release-gate principle.
- **[`fixtures.md`](fixtures.md)** — fixture requirements, layout, and the
  scaffold of fixtures committed under `fixtures/`.
- **[`parity.md`](parity.md)** — parity test types (loader, tokenizer,
  CPU/reference, GPU/CPU, cache, quantization) and parity reporting.
- **[`unsupported-configs.md`](unsupported-configs.md)** — required failure
  cases and the rule that unsupported features must fail explicitly, never
  fall back silently.
- **[`benchmarks.md`](benchmarks.md)** — when benchmarks come in (after M3),
  what they measure, and the rule that benchmarks never replace parity tests.

## Related Docs Outside This Directory

- [`../ci.md`](../ci.md) — CI policy, required PR checks, and the
  cross-milestone offline-by-default principle.
- [`../milestones/`](../milestones/) — per-milestone acceptance criteria;
  the test-matrix M1 traceability table maps these criteria to specific tests.
- [`../tasks/`](../tasks/) — per-milestone task breakdown with `Done when`
  conditions for each task.
