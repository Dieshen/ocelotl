use std::{
    fs::{File, remove_file},
    hint::black_box,
    io::Write,
    path::{Path, PathBuf},
};

use criterion::{Criterion, criterion_group, criterion_main};
use ocelotl_loader::{inspect_safetensors, load_safetensors_tensor_f32};

struct SyntheticSafetensors {
    path: PathBuf,
}

impl SyntheticSafetensors {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "ocelotl-loader-benchmark-{}.safetensors",
            std::process::id()
        ));
        write_synthetic_safetensors(&path);
        Self { path }
    }
}

impl Drop for SyntheticSafetensors {
    fn drop(&mut self) {
        let _ = remove_file(&self.path);
    }
}

fn benchmark_loader_artifacts(criterion: &mut Criterion) {
    let fixture = SyntheticSafetensors::new();

    criterion.bench_function("loader/safetensors/inspect/256x256_f32", |bencher| {
        bencher.iter(|| {
            let manifest = inspect_safetensors(black_box(&fixture.path))
                .expect("synthetic safetensors header must inspect");
            black_box(manifest);
        });
    });

    criterion.bench_function("loader/safetensors/load_f32/256x256", |bencher| {
        bencher.iter(|| {
            let tensor = load_safetensors_tensor_f32(black_box(&fixture.path), "weight")
                .expect("synthetic safetensors values must load");
            black_box(tensor);
        });
    });
}

fn write_synthetic_safetensors(path: &Path) {
    const ELEMENTS: usize = 256 * 256;
    let header = serde_json::json!({
        "weight": {
            "dtype": "F32",
            "shape": [256, 256],
            "data_offsets": [0, ELEMENTS * size_of::<f32>()]
        }
    });
    let header = serde_json::to_vec(&header).expect("serialize synthetic safetensors header");
    let mut file = File::create(path).expect("create synthetic safetensors benchmark fixture");
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("write safetensors header length");
    file.write_all(&header).expect("write safetensors header");
    for index in 0..ELEMENTS {
        let value = ((index % 127) as f32 - 63.0) / 127.0;
        file.write_all(&value.to_le_bytes())
            .expect("write deterministic safetensors payload");
    }
}

criterion_group!(benches, benchmark_loader_artifacts);
criterion_main!(benches);
