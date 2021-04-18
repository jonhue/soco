use crate::problem::types::{
    HomProblem, Schedule,
};

impl<'a, T> HomProblem<'a, T>
where
    T: Copy,
{
    fn _stream(
        &self,
        w: i32,
        alg: impl Fn(&HomProblem<'a, T>, i32, &Schedule<T>) -> T,
        next: impl Fn(&HomProblem<'a, T>, &Schedule<T>) -> Option<HomProblem<'a, T>>,
    ) -> Schedule<T> {
        let mut xs = vec![];
        let mut p = self;

        let mut tmp;
        loop {
            assert!(
                p.t_end > w + xs.len() as i32,
                "online problem must contain information for the next iteration"
            );

            xs.push(alg(p, w, &xs));
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
