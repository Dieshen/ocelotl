# Logit Fixtures

Keep only small reference logits or next-token outputs here. Do not commit large
model weights. Each fixture must name the source tool, model revision, prompt,
tolerance, and regeneration command.

Gemma4 fixtures that cover greedy decode should also include
`expected_decode_token`, derived as the argmax of the pinned final-token logits.

`gemma4_q4_k_m_basic_prompt_logits_reference.json` is a schema/command fixture
for an ignored real-artifact proof. It does not commit the full 262k-logit
reference dump; the ignored test generates that dump locally with llama.cpp
`llama-debug --save-logits` and compares every final-position logit at runtime.
