use std::collections::VecDeque;

use ocelotl_core::{InvalidRequestError, OcelotlError, Result, RuntimeError, TokenId};
use ocelotl_models::Qwen2_5Model;

use crate::decode_one_token;

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

pub struct QwenGreedyModel<'a> {
    model: &'a Qwen2_5Model,
}

impl<'a> QwenGreedyModel<'a> {
    pub fn new(model: &'a Qwen2_5Model) -> Self {
        Self { model }
    }
}

impl GreedyDecodeModel for QwenGreedyModel<'_> {
    fn decode_one(&self, prompt_tokens: &[TokenId]) -> Result<TokenId> {
        decode_one_token(self.model, prompt_tokens)
    }
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
            pending: VecDeque::new(),
            active: VecDeque::new(),
            completed: Vec::new(),
            cleanup_log: Vec::new(),
            events: Vec::new(),
        }
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

        let slot = RequestSlot {
            request_id: request.request_id,
            prompt_tokens: request.prompt_tokens,
            generated_tokens: Vec::with_capacity(request.max_new_tokens),
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
                    tokens: slot.generated_tokens.clone(),
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

        self.completed.sort_by_key(|response| {
            self.events
                .iter()
                .position(|event| event.request_id == response.request_id)
                .unwrap_or(usize::MAX)
        });
        Ok(self.completed.clone())
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
}

pub fn generate_qwen_batch(
    model: &Qwen2_5Model,
    requests: Vec<ScheduledGenerationRequest>,
    config: SchedulerConfig,
) -> Result<Vec<ScheduledGenerationResponse>> {
    let mut scheduler = ContinuousBatchScheduler::new(config);
    for request in requests {
        scheduler.submit(request)?;
    }
    scheduler.run_to_completion(&QwenGreedyModel::new(model))
}

fn runtime_err(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Runtime(RuntimeError {
        message: message.into(),
    })
}

#[cfg(test)]
mod tests {
    use ocelotl_core::DType;
    use ocelotl_models::{Qwen2_5Config, Qwen2_5LayerWeights, Qwen2_5Weights, transpose_2d};

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

    fn tiny_qwen_model() -> Qwen2_5Model {
        let cfg = Qwen2_5Config {
            vocab_size: 8,
            num_hidden_layers: 1,
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            context_length: 16,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::F32,
        };
        let h = cfg.hidden_size;
        let v = cfg.vocab_size;
        let q_out = cfg.num_attention_heads * cfg.head_dim;
        let kv_out = cfg.num_key_value_heads * cfg.head_dim;
        let i_size = cfg.intermediate_size;
        let embed: Vec<f32> = (0..v * h).map(|i| (i as f32) * 0.01).collect();
        let lm_head_w = transpose_2d(&embed, v, h);
        let weights = Qwen2_5Weights {
            embed_tokens: embed,
            layers: vec![Qwen2_5LayerWeights {
                q_proj_w: vec![0.01; h * q_out],
                q_proj_b: vec![0.0; q_out],
                k_proj_w: vec![0.01; h * kv_out],
                k_proj_b: vec![0.0; kv_out],
                v_proj_w: vec![0.01; h * kv_out],
                v_proj_b: vec![0.0; kv_out],
                o_proj_w: vec![0.01; q_out * h],
                input_layernorm_w: vec![1.0; h],
                post_attention_layernorm_w: vec![1.0; h],
                gate_proj_w: vec![0.01; h * i_size],
                up_proj_w: vec![0.01; h * i_size],
                down_proj_w: vec![0.01; i_size * h],
            }],
            final_norm_w: vec![1.0; h],
            lm_head_w,
            tie_word_embeddings: true,
        };
        Qwen2_5Model::new(cfg, weights).expect("tiny model must construct")
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
    fn qwen_batched_generation_matches_unbatched_decode() {
        let model = tiny_qwen_model();
        let requests = vec![request(1, &[1, 2], 1), request(2, &[2, 3], 1)];
        let expected: Vec<_> = requests
            .iter()
            .map(|req| ScheduledGenerationResponse {
                request_id: req.request_id,
                tokens: vec![decode_one_token(&model, &req.prompt_tokens).unwrap()],
            })
            .collect();

        let actual = generate_qwen_batch(&model, requests, SchedulerConfig { max_queue_len: 4 })
            .expect("batched generation must succeed");

        assert_eq!(actual, expected);
    }
}
