# Benchmark Fixtures

These fixtures describe benchmark harness shape only. They are small committed
JSON examples used by default tests to validate schema, truthful command
configuration, repeated alternating samples, aggregate statistics, environment
metadata, output comparability, and skip-record fields without running local
model artifacts or whisper.cpp.

The version 2 completed record is illustrative; its small three-measurement
plan represents a command-line override. The committed manifest defaults to ten
measured iterations for real local comparisons.

Real benchmark outputs belong under `local-artifacts/benchmarks/` and must not
be committed.
