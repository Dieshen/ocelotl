//! Device-tensor handle.
//!
//! `DeviceTensor` is the opaque handle that lets the model forward path chain
//! GPU primitives without bouncing each intermediate through host `Vec<f32>`.
//! CPU backends represent it as `Vec<f32>` under a `Mutex`; GPU backends use a
//! backend-private `DeviceBuffer` trait object so the kernels crate does not
//! pull GPU runtime types into its public API.
//!
//! # Variants
//!
//! - `Host` — backing is a `Vec<f32>` behind a `Mutex`. Used by the CPU
//!   backend and by any backend that can't yet keep this tensor on the
//!   device. Cheap host slice borrow via `borrow_host_slice`.
//! - `Device` — backing is a `Box<dyn DeviceBuffer>`. Reading as host requires
//!   the backend to perform a readback (`DeviceBuffer::to_host`).
//! - `View` — a `(source, offset, len)` triple over another `DeviceTensor`.
//!   This is the explicit "borrow a sub-range of a sibling tensor" variant
//!   needed by `attention_with_precomputed_kv` and the KV cache append story.
//!   Views are read-only.
//!
//! # Mutation through `&self`
//!
//! `KernelBackend::linear_d` takes the output handle as `&DeviceTensor`, not
//! `&mut DeviceTensor`, because multiple call sites need to share a buffer
//! (the encoder scratch pool, the cross-attention cache). Interior mutability
//! lives inside the variant:
//!
//! - `Host` uses `Mutex<Vec<f32>>`. Single-request contention is nil, so the
//!   lock overhead is uncontested.
//! - `Device` delegates mutation to the backend's buffer (CubeCL handles
//!   already provide interior mutability through their client API).
//! - `View` rejects mutation explicitly.

use std::{
    fmt::Debug,
    ops::Deref,
    sync::{Arc, Mutex, MutexGuard},
};

use ocelotl_core::{KernelError, OcelotlError, Result};

/// Backend-private handle to a GPU/accelerator buffer. The kernels crate
/// never inspects the contents — implementations live in the GPU backends
/// (`CubeClKernelBackend`, future CUDA/Metal/etc.).
pub trait DeviceBuffer: Send + Sync + Debug {
    /// Stable identifier for the backend that owns this buffer (e.g.
    /// `"cubecl-wgpu"`). Used for typed error messages.
    fn backend_id(&self) -> &'static str;

    /// Number of `f32` elements (not bytes).
    fn len_f32(&self) -> usize;

    /// Force a host readback. Allocates a `Vec<f32>` of length `len_f32()`
    /// and copies the device contents into it. Backends that already cache
    /// a host mirror may avoid a real transfer.
    fn to_host(&self) -> Result<Vec<f32>>;

    /// Overwrite the buffer contents from a host slice. `src.len()` must
    /// equal `len_f32()`.
    fn write_from_host(&self, src: &[f32]) -> Result<()>;

    /// Type-erased downcast hatch. GPU backends implement this so that
    /// in-backend overrides (e.g. `CubeClKernelBackend::linear_d`) can
    /// recognise their own buffers and run a device-resident launch
    /// without a host round-trip. The default returns `&self as &dyn Any`
    /// so the kernels crate doesn't need a `dyn Any` supertrait bound.
    fn as_any(&self) -> &dyn std::any::Any;
}

#[derive(Debug)]
enum DeviceTensorInner {
    /// Plain host-resident buffer.
    Host(Mutex<Vec<f32>>),
    /// Buffer that lives on a backend device (GPU, etc.).
    Device { buf: Box<dyn DeviceBuffer> },
    /// Read-only sub-range of another `DeviceTensor`.
    View {
        source: DeviceTensor,
        offset: usize,
        len: usize,
    },
}

/// Opaque, refcounted handle to an `f32` tensor that may live on host or
/// device. Cheap to clone (shares ownership of the underlying buffer).
#[derive(Debug, Clone)]
pub struct DeviceTensor {
    inner: Arc<DeviceTensorInner>,
}

impl DeviceTensor {
    /// Wrap an owned host `Vec<f32>` as a `DeviceTensor`. Zero-copy.
    pub fn from_host(v: Vec<f32>) -> Self {
        Self {
            inner: Arc::new(DeviceTensorInner::Host(Mutex::new(v))),
        }
    }

    /// Allocate a new host-resident tensor filled with zeros.
    pub fn host_zeros(len: usize) -> Self {
        Self::from_host(vec![0.0_f32; len])
    }

    /// Wrap a backend-owned device buffer.
    pub fn from_device(buf: Box<dyn DeviceBuffer>) -> Self {
        Self {
            inner: Arc::new(DeviceTensorInner::Device { buf }),
        }
    }

    /// Create a read-only view over a sub-range `offset..offset+len` of
    /// `source`. Errors if the range is out of bounds or if `source` is
    /// itself a view (views must be flattened to a single hop).
    pub fn view(source: &DeviceTensor, offset: usize, len: usize) -> Result<Self> {
        let src_len = source.len();
        if offset.checked_add(len).is_none_or(|end| end > src_len) {
            return Err(tensor_err(format!(
                "DeviceTensor::view range {offset}..{} exceeds source length {src_len}",
                offset.saturating_add(len)
            )));
        }
        if matches!(source.inner.as_ref(), DeviceTensorInner::View { .. }) {
            return Err(tensor_err(
                "DeviceTensor::view does not nest; resolve the source view first",
            ));
        }
        Ok(Self {
            inner: Arc::new(DeviceTensorInner::View {
                source: source.clone(),
                offset,
                len,
            }),
        })
    }

    /// Length in `f32` elements.
    pub fn len(&self) -> usize {
        match self.inner.as_ref() {
            DeviceTensorInner::Host(lock) => {
                lock.lock().expect("DeviceTensor host mutex poisoned").len()
            }
            DeviceTensorInner::Device { buf } => buf.len_f32(),
            DeviceTensorInner::View { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow a `&[f32]` slice if the backing is already host-resident or a
    /// view over host-resident memory. Errors for `Device` (caller must
    /// `to_host_owned` first to make the readback cost explicit).
    pub fn borrow_host_slice(&self) -> Result<HostBorrow<'_>> {
        match self.inner.as_ref() {
            DeviceTensorInner::Host(lock) => Ok(HostBorrow::Owned(
                lock.lock().expect("DeviceTensor host mutex poisoned"),
                0,
                None,
            )),
            DeviceTensorInner::View {
                source,
                offset,
                len,
            } => match source.inner.as_ref() {
                DeviceTensorInner::Host(lock) => Ok(HostBorrow::Owned(
                    lock.lock().expect("DeviceTensor host mutex poisoned"),
                    *offset,
                    Some(*len),
                )),
                _ => Err(tensor_err(
                    "DeviceTensor::borrow_host_slice: view source is not host-resident",
                )),
            },
            DeviceTensorInner::Device { buf } => Err(tensor_err(format!(
                "DeviceTensor::borrow_host_slice: backing is device-resident on {}; call to_host_owned() instead",
                buf.backend_id()
            ))),
        }
    }

    /// Borrow a `&mut [f32]` slice if the backing is host-resident. Errors
    /// for `Device` (use `write_from_host_slice` to push) or `View`
    /// (read-only).
    pub fn borrow_host_slice_mut(&self) -> Result<HostBorrowMut<'_>> {
        match self.inner.as_ref() {
            DeviceTensorInner::Host(lock) => Ok(HostBorrowMut {
                guard: lock.lock().expect("DeviceTensor host mutex poisoned"),
            }),
            DeviceTensorInner::View { .. } => Err(tensor_err(
                "DeviceTensor::borrow_host_slice_mut: views are read-only",
            )),
            DeviceTensorInner::Device { buf } => Err(tensor_err(format!(
                "DeviceTensor::borrow_host_slice_mut: backing is device-resident on {}; \
                 call write_from_host_slice() instead",
                buf.backend_id()
            ))),
        }
    }

    /// Always-owned host copy. `Host`: clones the Vec. `Device`: calls
    /// `DeviceBuffer::to_host` (readback). `View`: clones from the source.
    /// Costs a host allocation unconditionally — use sparingly.
    pub fn to_host_owned(&self) -> Result<Vec<f32>> {
        match self.inner.as_ref() {
            DeviceTensorInner::Host(lock) => Ok(lock
                .lock()
                .expect("DeviceTensor host mutex poisoned")
                .clone()),
            DeviceTensorInner::Device { buf } => buf.to_host(),
            DeviceTensorInner::View {
                source,
                offset,
                len,
            } => {
                let parent = source.to_host_owned()?;
                Ok(parent[*offset..*offset + *len].to_vec())
            }
        }
    }

    /// Copy `src` into the tensor's backing. `src.len()` must equal
    /// `self.len()`. Errors for `View` (read-only).
    pub fn write_from_host_slice(&self, src: &[f32]) -> Result<()> {
        let expected = self.len();
        if src.len() != expected {
            return Err(tensor_err(format!(
                "DeviceTensor::write_from_host_slice: src.len()={} != tensor len {}",
                src.len(),
                expected
            )));
        }
        match self.inner.as_ref() {
            DeviceTensorInner::Host(lock) => {
                lock.lock()
                    .expect("DeviceTensor host mutex poisoned")
                    .copy_from_slice(src);
                Ok(())
            }
            DeviceTensorInner::Device { buf } => buf.write_from_host(src),
            DeviceTensorInner::View { .. } => Err(tensor_err(
                "DeviceTensor::write_from_host_slice: views are read-only",
            )),
        }
    }

    /// If the backing is a `Device` variant, borrow the underlying
    /// `DeviceBuffer`. Returns `None` for `Host` and `View` variants.
    /// Backend overrides use this to recognise their own buffer type via
    /// `DeviceBuffer::as_any` and skip the host round-trip.
    pub fn try_as_device_buffer(&self) -> Option<&dyn DeviceBuffer> {
        match self.inner.as_ref() {
            DeviceTensorInner::Device { buf } => Some(buf.as_ref()),
            DeviceTensorInner::Host(_) | DeviceTensorInner::View { .. } => None,
        }
    }

    /// Tag identifying whether this tensor is currently host- or device-
    /// resident. Useful for diagnostic logging and assertions; do not branch
    /// on this in hot paths.
    pub fn residency(&self) -> Residency {
        match self.inner.as_ref() {
            DeviceTensorInner::Host(_) => Residency::Host,
            DeviceTensorInner::Device { buf } => Residency::Device(buf.backend_id()),
            DeviceTensorInner::View { source, .. } => source.residency(),
        }
    }
}

/// Result of `DeviceTensor::residency`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    Host,
    Device(&'static str),
}

/// Guard returned by `DeviceTensor::borrow_host_slice`. Derefs to `&[f32]`.
/// Holds the host mutex for the duration of the borrow.
#[derive(Debug)]
pub enum HostBorrow<'a> {
    /// Full host slice (Host variant).
    Owned(MutexGuard<'a, Vec<f32>>, usize, Option<usize>),
}

impl<'a> Deref for HostBorrow<'a> {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        match self {
            HostBorrow::Owned(guard, offset, len) => match len {
                Some(len) => &guard[*offset..*offset + *len],
                None => &guard[..],
            },
        }
    }
}

/// Guard returned by `DeviceTensor::borrow_host_slice_mut`. DerefMut to
/// `&mut [f32]`. Holds the host mutex for the duration of the borrow.
#[derive(Debug)]
pub struct HostBorrowMut<'a> {
    guard: MutexGuard<'a, Vec<f32>>,
}

impl<'a> Deref for HostBorrowMut<'a> {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        &self.guard[..]
    }
}

impl<'a> std::ops::DerefMut for HostBorrowMut<'a> {
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut self.guard[..]
    }
}

fn tensor_err(message: impl Into<String>) -> OcelotlError {
    OcelotlError::Kernel(KernelError {
        backend: "tensor".to_string(),
        message: message.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_host_round_trips_through_borrow() {
        let t = DeviceTensor::from_host(vec![1.0_f32, 2.0, 3.0, 4.0]);
        assert_eq!(t.len(), 4);
        assert_eq!(t.residency(), Residency::Host);
        let borrow = t.borrow_host_slice().expect("host borrow must succeed");
        assert_eq!(&*borrow, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn host_zeros_is_zero_filled() {
        let t = DeviceTensor::host_zeros(5);
        let borrow = t.borrow_host_slice().unwrap();
        assert_eq!(&*borrow, &[0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn write_from_host_slice_overwrites_backing() {
        let t = DeviceTensor::host_zeros(3);
        t.write_from_host_slice(&[7.0, 8.0, 9.0]).unwrap();
        let borrow = t.borrow_host_slice().unwrap();
        assert_eq!(&*borrow, &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn write_from_host_slice_rejects_length_mismatch() {
        let t = DeviceTensor::host_zeros(3);
        let err = t
            .write_from_host_slice(&[1.0, 2.0])
            .expect_err("must reject");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn view_borrows_a_subrange_of_host_source() {
        let source = DeviceTensor::from_host(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let v = DeviceTensor::view(&source, 1, 3).unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v.residency(), Residency::Host);
        let borrow = v.borrow_host_slice().unwrap();
        assert_eq!(&*borrow, &[20.0, 30.0, 40.0]);
    }

    #[test]
    fn view_rejects_out_of_bounds_range() {
        let source = DeviceTensor::from_host(vec![1.0, 2.0, 3.0]);
        let err = DeviceTensor::view(&source, 2, 2).expect_err("must reject");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn view_rejects_nested_views() {
        let source = DeviceTensor::from_host(vec![1.0, 2.0, 3.0, 4.0]);
        let v1 = DeviceTensor::view(&source, 0, 3).unwrap();
        let err = DeviceTensor::view(&v1, 0, 2).expect_err("nested view must be rejected");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn view_rejects_mutation() {
        let source = DeviceTensor::from_host(vec![1.0, 2.0, 3.0]);
        let v = DeviceTensor::view(&source, 0, 2).unwrap();
        let err = v
            .borrow_host_slice_mut()
            .expect_err("view must reject mut borrow");
        assert!(matches!(err, OcelotlError::Kernel(_)));
        let err = v
            .write_from_host_slice(&[9.0, 9.0])
            .expect_err("view must reject write");
        assert!(matches!(err, OcelotlError::Kernel(_)));
    }

    #[test]
    fn to_host_owned_returns_a_copy_of_host_data() {
        let t = DeviceTensor::from_host(vec![1.0, 2.0, 3.0]);
        let copy = t.to_host_owned().unwrap();
        assert_eq!(copy, vec![1.0, 2.0, 3.0]);
        // Mutate original; copy must not see it.
        t.write_from_host_slice(&[9.0, 9.0, 9.0]).unwrap();
        assert_eq!(copy, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn to_host_owned_through_view_copies_subrange() {
        let source = DeviceTensor::from_host(vec![10.0, 20.0, 30.0, 40.0]);
        let v = DeviceTensor::view(&source, 1, 2).unwrap();
        assert_eq!(v.to_host_owned().unwrap(), vec![20.0, 30.0]);
    }

    #[test]
    fn clone_shares_backing_storage() {
        let t1 = DeviceTensor::from_host(vec![1.0, 2.0]);
        let t2 = t1.clone();
        t1.write_from_host_slice(&[5.0, 6.0]).unwrap();
        let b = t2.borrow_host_slice().unwrap();
        assert_eq!(&*b, &[5.0, 6.0], "clone must share the same Mutex<Vec>");
    }
}
