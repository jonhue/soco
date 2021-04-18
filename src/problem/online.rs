use crate::problem::types::{HomProblem, Online, Schedule};

impl<'a, T> Online<HomProblem<'a, T>>
where
    T: Copy,
{
    fn _stream(
        &self,
        alg: impl Fn(&Online<HomProblem<'a, T>>, &Schedule<T>) -> T,
        next: impl Fn(
            &Online<HomProblem<'a, T>>,
            &Schedule<T>,
        ) -> Option<Online<HomProblem<'a, T>>>,
    ) -> Schedule<T> {
        let mut xs = vec![];
        let mut o = self;

        let mut tmp;
        loop {
            assert!(
                o.p.t_end > o.w + xs.len() as i32,
                "online problem must contain information for the next iteration"
            );

            xs.push(alg(o, &xs));
            o = match next(o, &xs) {
                None => break,
                Some(o) => {
                    tmp = o;
                    &tmp
                }
            };
        }

        xs
    }
}
