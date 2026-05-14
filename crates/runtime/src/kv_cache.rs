use std::collections::VecDeque;

use ocelotl_core::{
    DType, Device, KvCacheLayout, KvCacheStore, OcelotlError, PagedKvCacheLayout, Result,
    RuntimeError,
};
use ocelotl_models::qwen::Qwen2_5Config;

#[derive(Debug, Clone)]
pub struct ContiguousKvCache {
    layout: KvCacheLayout,
    keys: Vec<f32>,
    values: Vec<f32>,
    len_tokens: usize,
    released: bool,
}

impl ContiguousKvCache {
    pub fn new(layout: KvCacheLayout) -> Result<Self> {
        validate_cpu_f32_layout(&layout)?;
        let values = layout.values_per_all_key_tensors()?;
        Ok(Self {
            layout,
            keys: vec![0.0; values],
            values: vec![0.0; values],
            len_tokens: 0,
            released: false,
        })
    }

    pub fn for_qwen2_5(config: &Qwen2_5Config) -> Result<Self> {
        let layout = qwen2_5_kv_layout(config, config.context_length)?;
        Self::new(layout)
    }

    pub fn is_released(&self) -> bool {
        self.released
    }

    pub fn release(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.len_tokens = 0;
        self.released = true;
    }

    pub fn key_at(&self, layer: usize, position: usize) -> Result<&[f32]> {
        self.ensure_active()?;
        let width = self.layout.values_per_position()?;
        let start = self.layout.layer_position_offset(layer, position)?;
        Ok(&self.keys[start..start + width])
    }

    pub fn value_at(&self, layer: usize, position: usize) -> Result<&[f32]> {
        self.ensure_active()?;
        let width = self.layout.values_per_position()?;
        let start = self.layout.layer_position_offset(layer, position)?;
        Ok(&self.values[start..start + width])
    }

    fn ensure_active(&self) -> Result<()> {
        if self.released {
            Err(runtime_err("contiguous kv cache has been released"))
        } else {
            Ok(())
        }
    }
}

impl KvCacheStore for ContiguousKvCache {
    fn layout(&self) -> &KvCacheLayout {
        &self.layout
    }

    fn len_tokens(&self) -> usize {
        self.len_tokens
    }

    fn set_len_tokens(&mut self, len_tokens: usize) -> Result<()> {
        self.ensure_active()?;
        if len_tokens > self.layout.capacity_tokens {
            return Err(runtime_err(format!(
                "contiguous kv len {len_tokens} exceeds capacity {}",
                self.layout.capacity_tokens
            )));
        }
        self.len_tokens = len_tokens;
        Ok(())
    }

    fn write_layer_kv(
        &mut self,
        layer: usize,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<()> {
        self.ensure_active()?;
        let width = self.layout.values_per_position()?;
        if key.len() != width || value.len() != width {
            return Err(runtime_err(format!(
                "contiguous kv write expected key/value length {width}, got {}/{}",
                key.len(),
                value.len()
            )));
        }
        let start = self.layout.layer_position_offset(layer, position)?;
        self.keys[start..start + width].copy_from_slice(key);
        self.values[start..start + width].copy_from_slice(value);
        Ok(())
    }

    fn read_layer_keys(&self, layer: usize, len_tokens: usize, out: &mut [f32]) -> Result<()> {
        self.ensure_active()?;
        copy_contiguous_layer(&self.layout, &self.keys, layer, len_tokens, "keys", out)
    }

    fn read_layer_values(&self, layer: usize, len_tokens: usize, out: &mut [f32]) -> Result<()> {
        self.ensure_active()?;
        copy_contiguous_layer(&self.layout, &self.values, layer, len_tokens, "values", out)
    }
}

#[derive(Debug, Clone)]
pub struct PagedKvCache {
    layout: PagedKvCacheLayout,
    page_table: Vec<usize>,
    keys: Vec<f32>,
    values: Vec<f32>,
    len_tokens: usize,
    released: bool,
}

impl PagedKvCache {
    pub fn new_with_page_table(layout: PagedKvCacheLayout, page_table: Vec<usize>) -> Result<Self> {
        validate_cpu_f32_layout(&layout.base)?;
        layout.validate_page_table(&page_table)?;
        let page_values = layout.values_per_page_tensor()?;
        let tensor_values = checked_product(
            "paged kv tensor values",
            &[layout.base.num_layers, page_table.len(), page_values],
        )?;
        Ok(Self {
            layout,
            page_table,
            keys: vec![0.0; tensor_values],
            values: vec![0.0; tensor_values],
            len_tokens: 0,
            released: false,
        })
    }

    pub fn layout(&self) -> &PagedKvCacheLayout {
        &self.layout
    }

    pub fn page_table(&self) -> &[usize] {
        &self.page_table
    }

    pub fn physical_page_for_position(&self, position: usize) -> Result<usize> {
        let (logical, _) = self.layout.logical_page_and_offset(position)?;
        self.page_table.get(logical).copied().ok_or_else(|| {
            runtime_err(format!(
                "missing page table entry for logical page {logical}"
            ))
        })
    }

    pub fn is_released(&self) -> bool {
        self.released
    }

    pub fn release(&mut self) -> Vec<usize> {
        self.keys.clear();
        self.values.clear();
        self.len_tokens = 0;
        self.released = true;
        std::mem::take(&mut self.page_table)
    }

    fn ensure_active(&self) -> Result<()> {
        if self.released {
            Err(runtime_err("paged kv cache has been released"))
        } else {
            Ok(())
        }
    }

    fn storage_offset(&self, layer: usize, position: usize) -> Result<usize> {
        if layer >= self.layout.base.num_layers {
            return Err(runtime_err(format!(
                "paged kv layer {layer} out of range for {} layers",
                self.layout.base.num_layers
            )));
        }
        let (logical_page, offset_in_page) = self.layout.logical_page_and_offset(position)?;
        if logical_page >= self.page_table.len() {
            return Err(runtime_err(format!(
                "paged kv logical page {logical_page} missing from page table"
            )));
        }
        let page_values = self.layout.values_per_page_tensor()?;
        let layer_base = checked_product(
            "paged kv layer base",
            &[layer, self.page_table.len(), page_values],
        )?;
        let page_base = checked_product("paged kv page base", &[logical_page, page_values])?;
        let token_base = checked_product(
            "paged kv token base",
            &[offset_in_page, self.layout.base.values_per_position()?],
        )?;
        layer_base
            .checked_add(page_base)
            .and_then(|v| v.checked_add(token_base))
            .ok_or_else(|| runtime_err("paged kv storage offset overflowed usize"))
    }
}

impl KvCacheStore for PagedKvCache {
    fn layout(&self) -> &KvCacheLayout {
        &self.layout.base
    }

    fn len_tokens(&self) -> usize {
        self.len_tokens
    }

    fn set_len_tokens(&mut self, len_tokens: usize) -> Result<()> {
        self.ensure_active()?;
        if len_tokens > self.layout.base.capacity_tokens {
            return Err(runtime_err(format!(
                "paged kv len {len_tokens} exceeds capacity {}",
                self.layout.base.capacity_tokens
            )));
        }
        self.len_tokens = len_tokens;
        Ok(())
    }

    fn write_layer_kv(
        &mut self,
        layer: usize,
        position: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<()> {
        self.ensure_active()?;
        let width = self.layout.base.values_per_position()?;
        if key.len() != width || value.len() != width {
            return Err(runtime_err(format!(
                "paged kv write expected key/value length {width}, got {}/{}",
                key.len(),
                value.len()
            )));
        }
        let start = self.storage_offset(layer, position)?;
        self.keys[start..start + width].copy_from_slice(key);
        self.values[start..start + width].copy_from_slice(value);
        Ok(())
    }

    fn read_layer_keys(&self, layer: usize, len_tokens: usize, out: &mut [f32]) -> Result<()> {
        self.ensure_active()?;
        copy_paged_layer(self, &self.keys, layer, len_tokens, "keys", out)
    }

    fn read_layer_values(&self, layer: usize, len_tokens: usize, out: &mut [f32]) -> Result<()> {
        self.ensure_active()?;
        copy_paged_layer(self, &self.values, layer, len_tokens, "values", out)
    }
}

#[derive(Debug, Clone)]
pub struct PagedKvCacheAllocator {
    layout: PagedKvCacheLayout,
    free_pages: VecDeque<usize>,
}

impl PagedKvCacheAllocator {
    pub fn new(layout: PagedKvCacheLayout) -> Result<Self> {
        validate_cpu_f32_layout(&layout.base)?;
        Ok(Self {
            free_pages: (0..layout.physical_pages).collect(),
            layout,
        })
    }

    pub fn for_qwen2_5(
        config: &Qwen2_5Config,
        capacity_tokens: usize,
        page_size_tokens: usize,
        physical_pages: usize,
    ) -> Result<Self> {
        let base = qwen2_5_kv_layout(config, capacity_tokens)?;
        Self::new(PagedKvCacheLayout::new(
            base,
            page_size_tokens,
            physical_pages,
        )?)
    }

    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }

    pub fn allocate(&mut self, capacity_tokens: usize) -> Result<PagedKvCache> {
        if capacity_tokens == 0 || capacity_tokens > self.layout.base.capacity_tokens {
            return Err(runtime_err(format!(
                "paged kv requested capacity {capacity_tokens} outside allocator capacity {}",
                self.layout.base.capacity_tokens
            )));
        }
        let required = self.layout.required_pages_for_tokens(capacity_tokens)?;
        if required > self.free_pages.len() {
            return Err(runtime_err(format!(
                "paged kv allocation needs {required} pages but only {} are free",
                self.free_pages.len()
            )));
        }

        let mut table = Vec::with_capacity(required);
        for _ in 0..required {
            let page = self
                .free_pages
                .pop_front()
                .ok_or_else(|| runtime_err("paged kv free-page queue underflowed"))?;
            table.push(page);
        }

        let mut base = self.layout.base.clone();
        base.capacity_tokens = capacity_tokens;
        let cache_layout = PagedKvCacheLayout::new(
            base,
            self.layout.page_size_tokens,
            self.layout.physical_pages,
        )?;

        match PagedKvCache::new_with_page_table(cache_layout, table.clone()) {
            Ok(cache) => Ok(cache),
            Err(err) => {
                // Put pages back if cache construction fails after allocation.
                // Sorting keeps deterministic reuse behavior in tests.
                for page in table {
                    self.free_pages.push_back(page);
                }
                self.free_pages.make_contiguous().sort_unstable();
                Err(err)
            }
        }
    }

    pub fn release(&mut self, mut cache: PagedKvCache) {
        for page in cache.release() {
            self.free_pages.push_back(page);
        }
        self.free_pages.make_contiguous().sort_unstable();
    }
}

pub fn qwen2_5_kv_layout(config: &Qwen2_5Config, capacity_tokens: usize) -> Result<KvCacheLayout> {
    KvCacheLayout::new(
        config.num_hidden_layers,
        config.num_key_value_heads,
        capacity_tokens,
        config.head_dim,
        DType::F32,
        Device::Cpu,
    )
}

fn copy_contiguous_layer(
    layout: &KvCacheLayout,
    src: &[f32],
    layer: usize,
    len_tokens: usize,
    label: &str,
    out: &mut [f32],
) -> Result<()> {
    if len_tokens > layout.capacity_tokens {
        return Err(runtime_err(format!(
            "contiguous kv read len {len_tokens} exceeds capacity {}",
            layout.capacity_tokens
        )));
    }
    let width = layout.values_per_position()?;
    let expected = checked_product("contiguous kv read output", &[len_tokens, width])?;
    if out.len() != expected {
        return Err(runtime_err(format!(
            "contiguous kv read {label} expected output length {expected}, got {}",
            out.len()
        )));
    }
    for position in 0..len_tokens {
        let src_start = layout.layer_position_offset(layer, position)?;
        let dst_start = position * width;
        out[dst_start..dst_start + width].copy_from_slice(&src[src_start..src_start + width]);
    }
    Ok(())
}

fn copy_paged_layer(
    cache: &PagedKvCache,
    src: &[f32],
    layer: usize,
    len_tokens: usize,
    label: &str,
    out: &mut [f32],
) -> Result<()> {
    if len_tokens > cache.layout.base.capacity_tokens {
        return Err(runtime_err(format!(
            "paged kv read len {len_tokens} exceeds capacity {}",
            cache.layout.base.capacity_tokens
        )));
    }
    let width = cache.layout.base.values_per_position()?;
    let expected = checked_product("paged kv read output", &[len_tokens, width])?;
    if out.len() != expected {
        return Err(runtime_err(format!(
            "paged kv read {label} expected output length {expected}, got {}",
            out.len()
        )));
    }
    for position in 0..len_tokens {
        let src_start = cache.storage_offset(layer, position)?;
        let dst_start = position * width;
        out[dst_start..dst_start + width].copy_from_slice(&src[src_start..src_start + width]);
    }
    Ok(())
}

fn validate_cpu_f32_layout(layout: &KvCacheLayout) -> Result<()> {
    if layout.dtype != DType::F32 {
        return Err(runtime_err(format!(
            "runtime KV cache storage currently supports only F32, got {:?}",
            layout.dtype
        )));
    }
    if layout.device != Device::Cpu {
        return Err(runtime_err(format!(
            "runtime KV cache storage is CPU-resident in M5-M7, got {:?}",
            layout.device
        )));
    }
    Ok(())
}

fn checked_product(label: &str, dims: &[usize]) -> Result<usize> {
    dims.iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| runtime_err(format!("{label} overflowed usize for dims {dims:?}")))
}

fn runtime_err(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Runtime(RuntimeError {
        message: message.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> Qwen2_5Config {
        Qwen2_5Config {
            vocab_size: 32,
            num_hidden_layers: 2,
            hidden_size: 16,
            intermediate_size: 32,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            context_length: 8,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::F32,
        }
    }

    #[test]
    fn contiguous_cache_allocations_are_request_isolated() {
        let layout = qwen2_5_kv_layout(&tiny_cfg(), 4).unwrap();
        let mut a = ContiguousKvCache::new(layout.clone()).unwrap();
        let mut b = ContiguousKvCache::new(layout).unwrap();
        let key = vec![1.0; a.layout.values_per_position().unwrap()];
        let value = vec![2.0; a.layout.values_per_position().unwrap()];

        a.write_layer_kv(0, 0, &key, &value).unwrap();

        assert_eq!(a.key_at(0, 0).unwrap(), key);
        assert_ne!(b.key_at(0, 0).unwrap(), key);
        b.write_layer_kv(0, 0, &[3.0; 8], &[4.0; 8]).unwrap();
        assert_eq!(a.key_at(0, 0).unwrap(), key);
    }

    #[test]
    fn contiguous_cache_rejects_capacity_overflow_before_len_changes() {
        let layout = qwen2_5_kv_layout(&tiny_cfg(), 2).unwrap();
        let mut cache = ContiguousKvCache::new(layout).unwrap();

        let err = cache
            .set_len_tokens(3)
            .expect_err("len beyond capacity must be rejected");

        assert!(format!("{err}").contains("exceeds capacity"));
        assert_eq!(cache.len_tokens(), 0);
    }

    #[test]
    fn paged_allocator_releases_pages_back_to_pool() {
        let mut allocator = PagedKvCacheAllocator::for_qwen2_5(&tiny_cfg(), 8, 2, 4).unwrap();
        assert_eq!(allocator.free_page_count(), 4);

        let cache_a = allocator.allocate(4).unwrap();
        assert_eq!(cache_a.page_table(), &[0, 1]);
        let cache_b = allocator.allocate(2).unwrap();
        assert_eq!(cache_b.page_table(), &[2]);
        assert_eq!(allocator.free_page_count(), 1);

        allocator.release(cache_a);
        assert_eq!(allocator.free_page_count(), 3);
        let cache_c = allocator.allocate(4).unwrap();
        assert_eq!(cache_c.page_table(), &[0, 1]);
    }

    #[test]
    fn paged_cache_reads_and_writes_across_page_one() {
        let mut allocator = PagedKvCacheAllocator::for_qwen2_5(&tiny_cfg(), 6, 2, 4).unwrap();
        let mut cache = allocator.allocate(6).unwrap();
        let width = cache.layout().base.values_per_position().unwrap();

        for position in 0..5 {
            let key = vec![position as f32; width];
            let value = vec![100.0 + position as f32; width];
            cache.write_layer_kv(0, position, &key, &value).unwrap();
        }

        assert!(cache.physical_page_for_position(3).unwrap() > 0);
        let mut keys = vec![0.0; 5 * width];
        cache.read_layer_keys(0, 5, &mut keys).unwrap();
        assert_eq!(&keys[3 * width..4 * width], vec![3.0; width]);

        let layer_one_key = vec![7.0; width];
        let layer_one_value = vec![11.0; width];
        cache
            .write_layer_kv(1, 4, &layer_one_key, &layer_one_value)
            .unwrap();
        let mut layer_one_values = vec![0.0; 5 * width];
        cache
            .read_layer_values(1, 5, &mut layer_one_values)
            .unwrap();
        assert_eq!(&layer_one_values[4 * width..5 * width], vec![11.0; width]);
    }

    #[test]
    fn paged_cache_rejects_invalid_layouts_before_storage_use() {
        let mut base = qwen2_5_kv_layout(&tiny_cfg(), 4).unwrap();
        base.dtype = DType::BF16;
        let layout = PagedKvCacheLayout::new(base, 2, 2).unwrap();

        let err = PagedKvCache::new_with_page_table(layout, vec![0, 1])
            .expect_err("non-F32 paged cache must be rejected");

        assert!(format!("{err}").contains("F32"));
    }
}
