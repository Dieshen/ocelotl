# Fixtures

Fixtures make correctness repeatable. They should be small, deterministic, and
usable without network access.

## Fixture Requirements

Each fixture should document:

- Name.
- Purpose.
- Model family.
- Model artifact source and revision.
- Tokenizer source and revision.
- Prompt or input token IDs.
- Expected output.
- Tolerance.
- License considerations.

## Storage Policy

Small metadata and token fixtures can live in the repository. Large model files
should not be committed. If a fixture requires large artifacts, provide a script
or manifest later and keep normal tests offline by default.

## First Fixtures

M1 should start with synthetic or tiny hand-checked fixtures. M2 can add real
loader and tokenizer fixtures. M3 can add reference logits for one small model
path.

## Naming

Use descriptive fixture names:

```text
fixtures/tokenizer/llama3-basic-chat.json
fixtures/metadata/qwen2-tiny-metadata.json
fixtures/logits/m1-single-token-prefill.json
```

## Regeneration

If a fixture is generated from a tool, document the exact command and tool
version. Regenerating fixtures should be deliberate and reviewed.
