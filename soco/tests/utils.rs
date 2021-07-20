use std::collections::HashMap;

/// Constructs a hash map from a slice.
pub fn hash_map<K, V>(slice: &[(K, V)]) -> HashMap<K, V>
where
    K: Clone + Eq + std::hash::Hash + PartialEq,
    V: Clone,
{
    slice.iter().cloned().collect()
}

/// Selects upper bounds from decision space.
pub fn upper_bounds<T>(bounds: &Vec<(T, T)>) -> Vec<T>
where
    T: Copy,
{
    bounds.iter().map(|&(_, m)| m).collect()
}
