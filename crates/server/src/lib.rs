//! Server integration layer.

pub use ocelotl_core::GenerateResponse;
pub use ocelotl_runtime::Runtime;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerConfig {
    pub bind_addr: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:8080".to_string(),
        }
    }
}
