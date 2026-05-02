# M8 Tasks

M8 wraps runtime generation in the server layer. The server should expose stable,
intentional request, response, streaming, cancellation, and error-mapping
semantics without claiming broad API compatibility too early.

## Entry Criteria

- Runtime APIs can process generation requests and return deterministic outputs.
- Scheduler behavior is covered for the request lifecycle used by the server.
- Error taxonomy is stable enough for external mapping.

## Task List

- [ ] M8.1 Define server request and response DTOs.
  - Crates: `ocelotl-server`, `ocelotl-core`
  - Test first: deserialize valid and invalid JSON requests and assert the mapped runtime request or typed validation error.
  - Done when: server DTOs are separate from runtime internals but map cleanly into runtime contracts.

- [ ] M8.2 Add a mock runtime handle for server tests.
  - Crates: `ocelotl-server`
  - Test first: run handler tests against a fake runtime that returns fixed tokens and errors.
  - Done when: server behavior can be tested without loading a model.

- [ ] M8.3 Implement a minimal local generation endpoint.
  - Crates: `ocelotl-server`, `ocelotl-runtime`
  - Test first: send a local request to the handler and assert a complete non-streaming response.
  - Done when: endpoint uses runtime APIs and does not duplicate generation logic.

- [ ] M8.4 Map runtime errors to server responses.
  - Crates: `ocelotl-server`, `ocelotl-core`
  - Test first: feed invalid request, unsupported config, artifact error, overload, cancellation, and internal error cases through the handler.
  - Done when: status codes and response bodies are intentional, documented, and do not leak sensitive internals.

- [ ] M8.5 Implement streaming response lifecycle.
  - Crates: `ocelotl-server`, `ocelotl-runtime`
  - Test first: mock a token stream and assert token events, final event, and error event behavior.
  - Done when: streaming has clear completion and failure semantics.

- [ ] M8.6 Handle dropped clients and cancellation.
  - Crates: `ocelotl-server`, `ocelotl-runtime`
  - Test first: drop a streaming client and assert the runtime request is canceled and cleaned up.
  - Done when: server disconnects do not leak runtime work.

- [ ] M8.7 Add server validation and CI commands.
  - Crates: `ocelotl-server`, docs, CI
  - Test first: document the command that validates server tests before relying on it.
  - Done when: server validation is part of the milestone acceptance path and default CI if it remains offline.

- [ ] M8.8 Document compatibility boundaries.
  - Crates: docs only
  - Test first: public API docs state what is compatible, experimental, or intentionally unsupported.
  - Done when: server docs avoid accidental OpenAI/vLLM compatibility claims unless tests enforce them.

## Exit Criteria

- A minimal generation endpoint calls the runtime and returns deterministic responses in tests.
- Streaming lifecycle, dropped-client cancellation, and error mapping are tested.
- Server DTOs are separated from runtime internals.
- Public docs state compatibility boundaries clearly.

## Deferred

- Full OpenAI API compatibility.
- Authentication and multi-tenant policy.
- Distributed serving.
- Production deployment manifests.
