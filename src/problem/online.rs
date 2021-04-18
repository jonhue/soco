use crate::problem::types::{
    DiscreteHomProblem, DiscreteSchedule, HomProblem, Schedule,
};

impl<'a> DiscreteHomProblem<'a> {
    pub fn stream(
        &self,
        alg: impl Fn(&DiscreteHomProblem<'a>, i32) -> i32,
        next: impl Fn(
            &DiscreteHomProblem<'a>,
            &DiscreteSchedule,
        ) -> Option<DiscreteHomProblem<'a>>,
    ) -> DiscreteSchedule {
        self._stream(alg, 0, next)
    }
}

impl<'a, T> HomProblem<'a, T>
where
    T: Copy,
{
    fn _stream(
        &self,
        alg: impl Fn(&HomProblem<'a, T>, T) -> T,
        initial: T,
        next: impl Fn(&HomProblem<'a, T>, &Schedule<T>) -> Option<HomProblem<'a, T>>,
    ) -> Schedule<T> {
        let mut i = initial;
        let mut xs = vec![];
        let mut p = self;

        let mut tmp;
        loop {
            assert!(p.t_end > xs.len() as i32, "online problem must contain information for the next iteration");

            i = alg(p, i);
            xs.push(i);
            p = match next(p, &xs) {
                None => break,
                Some(p) => {
                    tmp = p;
                    &tmp
                }
            };
        }

        xs
    }
}
