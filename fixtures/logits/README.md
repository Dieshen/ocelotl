# Logit Fixtures

Keep only small reference logits or next-token outputs here. Do not commit large
model weights. Each fixture must name the source tool, model revision, prompt,
tolerance, and regeneration command.

Gemma4 fixtures that cover greedy decode should also include
`expected_decode_token`, derived as the argmax of the pinned final-token logits.
