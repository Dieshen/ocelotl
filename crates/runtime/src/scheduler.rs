//! Generic continuous-batch scheduler.
//!
//! Family-specific adapters (e.g. `QwenGreedyModel`, `generate_qwen_batch`)
//! live in the matching family module (`crate::qwen`), not here. Keep this
//! file generic over `GreedyDecodeModel`.

use std::collections::VecDeque;

use ocelotl_core::{
    InvalidRequestError, OcelotlError, RequestLimits, Result, RuntimeError, TokenId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    pub max_queue_len: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self { max_queue_len: 128 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerRequestState {
    Queued,
    Prefill,
    Decode,
    Emit,
    Complete,
    Canceled,
    Cleanup,
}

impl SchedulerRequestState {
    pub fn transition(self, next: Self) -> Result<Self> {
        let allowed = matches!(
            (self, next),
            (Self::Queued, Self::Prefill)
                | (Self::Prefill, Self::Decode)
                | (Self::Decode, Self::Emit)
                | (Self::Emit, Self::Decode)
                | (Self::Emit, Self::Complete)
                | (Self::Complete, Self::Cleanup)
                | (Self::Queued, Self::Canceled)
                | (Self::Prefill, Self::Canceled)
                | (Self::Decode, Self::Canceled)
                | (Self::Emit, Self::Canceled)
                | (Self::Canceled, Self::Cleanup)
        );
        if allowed {
            Ok(next)
        } else {
            Err(runtime_err(format!(
                "invalid scheduler transition {self:?} -> {next:?}"
            )))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledGenerationRequest {
    pub request_id: u64,
    pub prompt_tokens: Vec<TokenId>,
    pub max_new_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledGenerationResponse {
    pub request_id: u64,
    pub tokens: Vec<TokenId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerEvent {
    pub request_id: u64,
    pub state: SchedulerRequestState,
    pub token: Option<TokenId>,
}

pub trait GreedyDecodeModel {
    fn decode_one(&self, prompt_tokens: &[TokenId]) -> Result<TokenId>;
}

#[derive(Debug, Clone)]
struct RequestSlot {
    request_id: u64,
    prompt_tokens: Vec<TokenId>,
    generated_tokens: Vec<TokenId>,
    max_new_tokens: usize,
    state: SchedulerRequestState,
}

#[derive(Debug, Clone)]
pub struct ContinuousBatchScheduler {
    config: SchedulerConfig,
    limits: RequestLimits,
    pending: VecDeque<RequestSlot>,
    active: VecDeque<RequestSlot>,
    completed: Vec<ScheduledGenerationResponse>,
    cleanup_log: Vec<u64>,
    events: Vec<SchedulerEvent>,
}

impl ContinuousBatchScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            limits: RequestLimits::default(),
            pending: VecDeque::new(),
            active: VecDeque::new(),
            completed: Vec::new(),
            cleanup_log: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Construct a scheduler with deployment-specific request ceilings.
    pub fn with_limits(config: SchedulerConfig, limits: RequestLimits) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            config,
            limits,
            pending: VecDeque::new(),
            active: VecDeque::new(),
            completed: Vec::new(),
            cleanup_log: Vec::new(),
            events: Vec::new(),
        })
    }

    pub fn submit(&mut self, request: ScheduledGenerationRequest) -> Result<()> {
        if self.pending.len() + self.active.len() >= self.config.max_queue_len {
            return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                field: "scheduler.queue".to_string(),
                message: format!(
                    "queue is full at configured max_queue_len {}",
                    self.config.max_queue_len
                ),
            }));
        }
        if request.prompt_tokens.is_empty() {
            return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                field: "prompt_tokens".to_string(),
                message: "must contain at least one token".to_string(),
            }));
        }
        if request.max_new_tokens == 0 {
            return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                field: "max_new_tokens".to_string(),
                message: "must be greater than zero".to_string(),
            }));
        }
        self.limits
            .validate_generation(request.prompt_tokens.len(), request.max_new_tokens)?;
        if self
            .pending
            .iter()
            .chain(self.active.iter())
            .any(|slot| slot.request_id == request.request_id)
            || self
                .completed
                .iter()
                .any(|response| response.request_id == request.request_id)
        {
            return Err(OcelotlError::InvalidRequest(InvalidRequestError {
                field: "request_id".to_string(),
                message: format!("request id {} is already scheduled", request.request_id),
            }));
        }

        let mut generated_tokens = Vec::new();
        generated_tokens
            .try_reserve_exact(request.max_new_tokens)
            .map_err(|source| {
                OcelotlError::Runtime(RuntimeError {
                    message: format!(
                        "failed to reserve {} generated tokens: {source}",
                        request.max_new_tokens
                    ),
                })
            })?;

        self.start_new_batch_history_if_idle();
        let slot = RequestSlot {
            request_id: request.request_id,
            prompt_tokens: request.prompt_tokens,
            generated_tokens,
            max_new_tokens: request.max_new_tokens,
            state: SchedulerRequestState::Queued,
        };
        self.events.push(SchedulerEvent {
            request_id: slot.request_id,
            state: slot.state,
            token: None,
        });
        self.pending.push_back(slot);
        Ok(())
    }

    pub fn cancel(&mut self, request_id: u64) -> Result<()> {
        if let Some(idx) = self.pending.iter().position(|r| r.request_id == request_id) {
            let mut slot = self.pending.remove(idx).expect("index came from position");
            slot.state = slot.state.transition(SchedulerRequestState::Canceled)?;
            self.events.push(SchedulerEvent {
                request_id,
                state: slot.state,
                token: None,
            });
            self.cleanup(slot)?;
            return Ok(());
        }
        if let Some(slot) = self.active.iter_mut().find(|r| r.request_id == request_id) {
            slot.state = slot.state.transition(SchedulerRequestState::Canceled)?;
            self.events.push(SchedulerEvent {
                request_id,
                state: slot.state,
                token: None,
            });
            return Ok(());
        }
        Err(runtime_err(format!(
            "request {request_id} is not pending or active"
        )))
    }

    pub fn run_to_completion<M: GreedyDecodeModel>(
        &mut self,
        model: &M,
    ) -> Result<Vec<ScheduledGenerationResponse>> {
        self.admit_pending()?;

        while let Some(mut slot) = self.active.pop_front() {
            if slot.state == SchedulerRequestState::Canceled {
                self.cleanup(slot)?;
                continue;
            }

            slot.state = slot.state.transition(SchedulerRequestState::Emit)?;
            let token = model.decode_one(&slot.prompt_tokens)?;
            slot.generated_tokens.push(token);
            slot.prompt_tokens.push(token);
            self.events.push(SchedulerEvent {
                request_id: slot.request_id,
                state: SchedulerRequestState::Emit,
                token: Some(token),
            });

            if slot.generated_tokens.len() == slot.max_new_tokens {
                slot.state = slot.state.transition(SchedulerRequestState::Complete)?;
                self.events.push(SchedulerEvent {
                    request_id: slot.request_id,
                    state: SchedulerRequestState::Complete,
                    token: None,
                });
                self.completed.push(ScheduledGenerationResponse {
                    request_id: slot.request_id,
                    tokens: std::mem::take(&mut slot.generated_tokens),
                });
                self.cleanup(slot)?;
            } else {
                slot.state = slot.state.transition(SchedulerRequestState::Decode)?;
                self.events.push(SchedulerEvent {
                    request_id: slot.request_id,
                    state: SchedulerRequestState::Decode,
                    token: None,
                });
                self.active.push_back(slot);
            }
        }

        let mut completed = std::mem::take(&mut self.completed);
        completed.sort_by_key(|response| {
            self.events
                .iter()
                .position(|event| event.request_id == response.request_id)
                .unwrap_or(usize::MAX)
        });
        Ok(completed)
    }

    pub fn cleanup_log(&self) -> &[u64] {
        &self.cleanup_log
    }

    pub fn events(&self) -> &[SchedulerEvent] {
        &self.events
    }

    fn admit_pending(&mut self) -> Result<()> {
        while let Some(mut slot) = self.pending.pop_front() {
            slot.state = slot.state.transition(SchedulerRequestState::Prefill)?;
            self.events.push(SchedulerEvent {
                request_id: slot.request_id,
                state: slot.state,
                token: None,
            });
            slot.state = slot.state.transition(SchedulerRequestState::Decode)?;
            self.events.push(SchedulerEvent {
                request_id: slot.request_id,
                state: slot.state,
                token: None,
            });
            self.active.push_back(slot);
        }
        Ok(())
    }

    fn cleanup(&mut self, mut slot: RequestSlot) -> Result<()> {
        slot.state = slot.state.transition(SchedulerRequestState::Cleanup)?;
        self.events.push(SchedulerEvent {
            request_id: slot.request_id,
            state: SchedulerRequestState::Cleanup,
            token: None,
        });
        self.cleanup_log.push(slot.request_id);
        Ok(())
    }

    fn start_new_batch_history_if_idle(&mut self) {
        if self.pending.is_empty() && self.active.is_empty() && self.completed.is_empty() {
            self.events.clear();
            self.cleanup_log.clear();
        }
    }
}

fn runtime_err(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Runtime(RuntimeError {
        message: message.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct IncrementingMockModel;

    impl GreedyDecodeModel for IncrementingMockModel {
        fn decode_one(&self, prompt_tokens: &[TokenId]) -> Result<TokenId> {
            Ok(TokenId(
                prompt_tokens.last().expect("non-empty prompt").0 + 1,
            ))
        }
    }

    fn request(id: u64, prompt: &[u32], max_new_tokens: usize) -> ScheduledGenerationRequest {
        ScheduledGenerationRequest {
            request_id: id,
            prompt_tokens: prompt.iter().copied().map(TokenId).collect(),
            max_new_tokens,
        }
    }

    #[test]
    fn state_transitions_reject_invalid_edges() {
        assert_eq!(
            SchedulerRequestState::Queued
                .transition(SchedulerRequestState::Prefill)
                .unwrap(),
            SchedulerRequestState::Prefill
        );

        let err = SchedulerRequestState::Cleanup
            .transition(SchedulerRequestState::Decode)
            .expect_err("cleanup cannot return to decode");

        assert!(format!("{err}").contains("invalid scheduler transition"));
    }

    #[test]
    fn scheduler_emits_tokens_round_robin_for_mock_requests() {
        let mut scheduler = ContinuousBatchScheduler::new(SchedulerConfig { max_queue_len: 4 });
        scheduler.submit(request(10, &[1], 2)).unwrap();
        scheduler.submit(request(20, &[100], 1)).unwrap();

        let responses = scheduler.run_to_completion(&IncrementingMockModel).unwrap();

        assert_eq!(
            responses,
            vec![
                ScheduledGenerationResponse {
                    request_id: 10,
                    tokens: vec![TokenId(2), TokenId(3)]
                },
                ScheduledGenerationResponse {
                    request_id: 20,
                    tokens: vec![TokenId(101)]
                },
            ]
        );
        let emitted: Vec<(u64, TokenId)> = scheduler
            .events()
            .iter()
            .filter_map(|event| event.token.map(|token| (event.request_id, token)))
            .collect();
        assert_eq!(
            emitted,
            vec![(10, TokenId(2)), (20, TokenId(101)), (10, TokenId(3))]
        );
    }

    #[test]
    fn scheduler_cancels_one_request_without_cleaning_active_peer() {
        let mut scheduler = ContinuousBatchScheduler::new(SchedulerConfig { max_queue_len: 4 });
        scheduler.submit(request(1, &[1], 1)).unwrap();
        scheduler.submit(request(2, &[2], 1)).unwrap();
        scheduler.cancel(1).unwrap();

        let responses = scheduler.run_to_completion(&IncrementingMockModel).unwrap();

        assert_eq!(
            responses,
            vec![ScheduledGenerationResponse {
                request_id: 2,
                tokens: vec![TokenId(3)]
            }]
        );
        assert_eq!(scheduler.cleanup_log(), &[1, 2]);
    }

    #[test]
    fn scheduler_rejects_requests_beyond_queue_bound() {
        let mut scheduler = ContinuousBatchScheduler::new(SchedulerConfig { max_queue_len: 1 });
        scheduler.submit(request(1, &[1], 1)).unwrap();

        let err = scheduler
            .submit(request(2, &[2], 1))
            .expect_err("bounded scheduler must reject excess requests");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "scheduler.queue");
                assert!(invalid.message.contains("queue is full"));
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn scheduler_rejects_duplicate_request_ids() {
        let mut scheduler = ContinuousBatchScheduler::new(SchedulerConfig { max_queue_len: 4 });
        scheduler.submit(request(1, &[1], 1)).unwrap();

        let err = scheduler
            .submit(request(1, &[2], 1))
            .expect_err("duplicate request ids make cancellation ambiguous");

        match err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "request_id");
                assert!(invalid.message.contains("already scheduled"));
            }
            other => panic!("expected InvalidRequest(request_id), got {other:?}"),
        }
    }

    #[test]
    fn scheduler_short_request_makes_progress_before_long_request_completes() {
        let mut scheduler = ContinuousBatchScheduler::new(SchedulerConfig { max_queue_len: 4 });
        scheduler.submit(request(1, &[1], 4)).unwrap();
        scheduler.submit(request(2, &[10], 1)).unwrap();

        scheduler.run_to_completion(&IncrementingMockModel).unwrap();

        let emitted: Vec<u64> = scheduler
            .events()
            .iter()
            .filter(|event| event.token.is_some())
            .map(|event| event.request_id)
            .collect();
        assert_eq!(emitted[0], 1);
        assert_eq!(emitted[1], 2);
        assert!(emitted[2..].iter().all(|id| *id == 1));
    }

    #[test]
    fn scheduler_with_limits_rejects_output_and_combined_context_before_reservation() {
        let limits = RequestLimits {
            max_prompt_tokens: 4,
            max_new_tokens: 2,
            max_context_tokens: 5,
            max_audio_samples: 16_000,
        };
        let mut scheduler =
            ContinuousBatchScheduler::with_limits(SchedulerConfig { max_queue_len: 4 }, limits)
                .expect("valid limits must construct a scheduler");

        let output_err = scheduler
            .submit(request(1, &[1], 3))
            .expect_err("max_new_tokens over the policy must fail");
        assert!(matches!(output_err, OcelotlError::InvalidRequest(_)));

        let context_err = scheduler
            .submit(request(2, &[1, 2, 3, 4], 2))
            .expect_err("prompt plus output over the context policy must fail");
        match context_err {
            OcelotlError::InvalidRequest(invalid) => {
                assert_eq!(invalid.field, "context_tokens");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn scheduler_rejects_pathological_reservation_without_panicking() {
        let mut scheduler = ContinuousBatchScheduler::new(SchedulerConfig { max_queue_len: 1 });

        let err = scheduler
            .submit(request(1, &[1], usize::MAX))
            .expect_err("pathological token capacity must return an error");

        assert!(matches!(err, OcelotlError::InvalidRequest(_)));
    }

    #[test]
    fn scheduler_reuse_returns_only_current_batch_and_replaces_history() {
        let mut scheduler = ContinuousBatchScheduler::new(SchedulerConfig { max_queue_len: 2 });
        scheduler.submit(request(1, &[1], 1)).unwrap();
        let first = scheduler.run_to_completion(&IncrementingMockModel).unwrap();
        assert_eq!(first[0].request_id, 1);

        scheduler.submit(request(2, &[10], 1)).unwrap();
        let second = scheduler.run_to_completion(&IncrementingMockModel).unwrap();

        assert_eq!(second.len(), 1);
        assert_eq!(second[0].request_id, 2);
        assert!(scheduler.events().iter().all(|event| event.request_id == 2));
        assert_eq!(scheduler.cleanup_log(), &[2]);

        scheduler
            .submit(request(1, &[100], 1))
            .expect("a completed request id may be reused in a later batch");
    }
}
