//! Utilities to use iterators with wrappers around vectors.

pub trait VecWrapper {
    type Item;

    fn to_vec(&self) -> &Vec<Self::Item>;

    fn iter(&self) -> Iter<Self::Item> {
        Iter(Box::new(self.to_vec().iter()))
    }
}

pub struct Iter<'a, T>(Box<dyn Iterator<Item = &'a T> + 'a>);

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
