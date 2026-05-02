# Milestone Task Backlog

This directory turns the milestone specs into executable task lists. The
milestone docs remain the design source of truth; these files are the working
backlog for test-first implementation.

## How To Use This Backlog

1. Read `docs/start-here.md`.
2. Read the relevant milestone spec under `docs/milestones/`.
3. Pick the next unchecked task from the matching task document.
4. Write the smallest failing test named by the task.
5. Implement only enough code to make that test pass.
6. Update the task doc if implementation reveals a better split or missing task.

Do not treat a task as complete because code exists. A task is complete when its
`Done when` condition is met and the validation commands for the milestone pass.

## Task Documents

| Milestone | Task Backlog | Milestone Spec |
| --- | --- | --- |
| M0 | `docs/tasks/m0-skeleton.md` | `docs/milestones/m0-skeleton.md` |
| M1 | `docs/tasks/m1-cpu-reference.md` | `docs/milestones/m1-cpu-reference.md` |
| M2 | `docs/tasks/m2-loader-tokenizer.md` | `docs/milestones/m2-loader-tokenizer.md` |
| M3 | `docs/tasks/m3-single-model-forward.md` | `docs/milestones/m3-single-model-forward.md` |
| M4 | `docs/tasks/m4-gpu-kernel-path.md` | `docs/milestones/m4-gpu-kernel-path.md` |
| M5 | `docs/tasks/m5-contiguous-kv-cache.md` | `docs/milestones/m5-contiguous-kv-cache.md` |
| M6 | `docs/tasks/m6-paged-kv-cache.md` | `docs/milestones/m6-paged-kv-cache.md` |
| M7 | `docs/tasks/m7-continuous-batching.md` | `docs/milestones/m7-continuous-batching.md` |
| M8 | `docs/tasks/m8-server-api.md` | `docs/milestones/m8-server-api.md` |

## Task Format

Each task should include:

- `Crates`: the crate or crates that own the change.
- `Test first`: the first failing test or fixture that should be added.
- `Done when`: the observable condition that completes the task.

Tasks may be split as work becomes clearer, but task documents should not drift
into design specs. Put design decisions in `docs/design/` or the relevant
milestone spec, then link back here.

## Project Rules

- Keep implementation test-driven.
- Keep default tests offline.
- Keep unsupported model features explicit and tested.
- Keep crate boundary violations out of task completion.
- Update validation docs when a milestone adds new required commands.
