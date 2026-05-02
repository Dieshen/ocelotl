# M7 Tasks

M7 introduces scheduling and continuous batching. The scheduler must preserve the
single-request correctness already proven by M1-M6.

## Entry Criteria

- Paged or contiguous KV cache behavior is correct for single requests.
- Runtime request lifecycle has explicit allocation, cancellation, and cleanup semantics.

## Task List

- [ ] M7.1 Build a scheduler test harness with a mock model.
  - Crates: `ocelotl-runtime`
  - Test first: enqueue two deterministic mock requests and assert the emitted token order.
  - Done when: scheduling behavior can be tested without loading a real model.

- [ ] M7.2 Define scheduler work item types.
  - Crates: `ocelotl-runtime`, `ocelotl-core`
  - Test first: validate transitions among prefill, decode, emit, complete, and cleanup states.
  - Done when: request state transitions are explicit and invalid transitions fail in tests.

- [ ] M7.3 Preserve batched/unbatched greedy parity.
  - Crates: `ocelotl-runtime`, `ocelotl-models`
  - Test first: run two requests independently and batched, then compare emitted token sequences.
  - Done when: batching changes throughput mechanics, not deterministic output.

- [ ] M7.4 Add cancellation behavior.
  - Crates: `ocelotl-runtime`
  - Test first: cancel one request while another remains active and assert only the canceled request is cleaned up.
  - Done when: cancellation does not corrupt active request state or cache ownership.

- [ ] M7.5 Add bounded queue and backpressure behavior.
  - Crates: `ocelotl-runtime`
  - Test first: fill the request queue and assert later requests receive a typed overload or backpressure result.
  - Done when: scheduler cannot grow memory unbounded under load.

- [ ] M7.6 Add fairness and starvation tests.
  - Crates: `ocelotl-runtime`
  - Test first: mix a long request and short request and assert the short request makes progress under the chosen policy.
  - Done when: fairness expectations are documented and tested.

- [ ] M7.7 Integrate scheduler with runtime generation APIs.
  - Crates: `ocelotl-runtime`
  - Test first: call the public runtime API for two concurrent requests and assert deterministic outputs.
  - Done when: runtime users do not need to call scheduler internals.

- [ ] M7.8 Document scheduler invariants.
  - Crates: docs only
  - Test first: scheduler implementation changes require updates to `docs/design/scheduler.md`.
  - Done when: queue limits, request states, cancellation, and fairness policy are documented.

## Exit Criteria

- Multiple requests can progress through prefill and decode scheduling.
- Batched outputs match unbatched deterministic outputs.
- Cancellation, cleanup, queue bounds, and fairness are tested.
- Scheduler internals remain behind runtime APIs.

## Deferred

- Distributed scheduling.
- Speculative decoding.
- Priority classes beyond the first fairness policy.
- Multi-model scheduling.
