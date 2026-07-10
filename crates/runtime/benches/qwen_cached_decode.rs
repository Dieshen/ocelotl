use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use ocelotl_core::{DType, TokenId};
use ocelotl_models::qwen::{
    Qwen2_5Config, Qwen2_5LayerWeights, Qwen2_5Model, Qwen2_5Weights, transpose_2d,
};
use ocelotl_runtime::qwen::{
    decode_one_token_with_contiguous_cache, prepare_qwen2_5_contiguous_cache,
};

fn benchmark_cached_decode(criterion: &mut Criterion) {
    let model = tiny_qwen_model();
    let initial_state =
        prepare_qwen2_5_contiguous_cache(&model, &[TokenId(1), TokenId(2), TokenId(3), TokenId(4)])
            .expect("tiny Qwen prompt must prepare a contiguous cache");

    criterion.bench_function("runtime/qwen2_5/cached_decode/context_4", |bencher| {
        bencher.iter_batched(
            || initial_state.clone(),
            |mut state| {
                let token = decode_one_token_with_contiguous_cache(
                    black_box(&model),
                    black_box(&mut state),
                )
                .expect("tiny cached decode must succeed");
                black_box(token);
            },
            BatchSize::SmallInput,
        );
    });
}

fn tiny_qwen_model() -> Qwen2_5Model {
    let config = Qwen2_5Config {
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
    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let query = config.num_attention_heads * config.head_dim;
    let key_value = config.num_key_value_heads * config.head_dim;
    let intermediate = config.intermediate_size;
    let embeddings = (0..vocab * hidden)
        .map(|index| index as f32 * 0.01)
        .collect::<Vec<_>>();
    let lm_head = transpose_2d(&embeddings, vocab, hidden);
    let weights = Qwen2_5Weights {
        embed_tokens: embeddings,
        layers: vec![Qwen2_5LayerWeights {
            q_proj_w: vec![0.01; hidden * query],
            q_proj_b: vec![0.0; query],
            k_proj_w: vec![0.01; hidden * key_value],
            k_proj_b: vec![0.0; key_value],
            v_proj_w: vec![0.01; hidden * key_value],
            v_proj_b: vec![0.0; key_value],
            o_proj_w: vec![0.01; query * hidden],
            input_layernorm_w: vec![1.0; hidden],
            post_attention_layernorm_w: vec![1.0; hidden],
            gate_proj_w: vec![0.01; hidden * intermediate],
            up_proj_w: vec![0.01; hidden * intermediate],
            down_proj_w: vec![0.01; intermediate * hidden],
        }],
        final_norm_w: vec![1.0; hidden],
        lm_head_w: lm_head,
        tie_word_embeddings: true,
    };
    Qwen2_5Model::new(config, weights).expect("tiny benchmark model must construct")
}

criterion_group!(benches, benchmark_cached_decode);
criterion_main!(benches);
