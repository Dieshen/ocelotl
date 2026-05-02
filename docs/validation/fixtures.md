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

## Existing Scaffold

The repository starts with fixture directories and synthetic placeholders:

- `fixtures/metadata/qwen2_5_tiny_synthetic.json`
- `fixtures/metadata/unsupported_unknown_architecture.json`
- `fixtures/tokenizer/qwen2_5_basic_prompt.json`
- `fixtures/logits/README.md`

M2 must replace tokenizer placeholder IDs with exact IDs from a pinned tokenizer
revision before treating that fixture as passing behavior.
