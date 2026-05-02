//! Root crate for the Ocelotl inference runtime workspace.

pub use ocelotl_core as core;
pub use ocelotl_runtime as runtime;

/// Current public crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
