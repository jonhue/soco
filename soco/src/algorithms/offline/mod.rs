//! Offline Algorithms.

pub struct OfflineOptions {
    /// Compute inverted cost.
    pub inverted: bool,
}

pub mod multi_dimensional;
pub mod uni_dimensional;
