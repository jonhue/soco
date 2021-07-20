use std::collections::HashMap;

/// Constructs a hash map from a slice.
pub fn hash_map<K, V>(slice: &[(K, V)]) -> HashMap<K, V>
where
    K: Clone + Eq + std::hash::Hash + PartialEq,
    V: Clone,
{
    slice.iter().cloned().collect()
}
