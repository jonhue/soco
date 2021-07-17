use std::{
    io::Write,
    net::{SocketAddr, TcpStream},
};

use crossbeam::thread;

use crate::{
    algorithms::{
        online::{Memory, OnlineAlgorithm},
        Options,
    },
    model::{Model, OfflineInput, OnlineInput},
    problem::{Online, Problem},
    result::Result,
    schedule::Schedule,
    value::Value,
};

use super::backend;

/// Starts backend server in a new thread.
/// Returns initial schedule and last memory of the algorithm.
pub fn start<'a, T, P, M, O, A, B>(
    addr: SocketAddr,
    model: &'a impl Model<'a, P, A, B>,
    alg: &'a impl OnlineAlgorithm<'a, T, P, M, O>,
    options: O,
    w: i32,
    input: A,
) -> Result<(Schedule<T>, Option<M>)>
where
    T: Value<'a> + Send,
    P: Problem + Send + 'a,
    M: Memory<'a, P> + Send,
    O: Options<P> + Send + 'a,
    A: OfflineInput,
    B: OnlineInput<'a>,
{
    let p = model.to(input);
    let t_end = p.t_end();
    let mut o = Online { p, w };

    // stream until current time slot
    let result = o.offline_stream(alg, t_end, options.clone())?;
    let (mut xs, prev_m) = result.clone();

    // start backend server
    thread::scope(move |s| {
        s.spawn(move |_| {
            backend::start(addr, model, o, alg, &mut xs, prev_m, options);
        });
    })
    .unwrap();

    Ok(result)
}

/// Executes next iteration of online algorithm.
pub fn next<'a>(addr: SocketAddr, input: impl OnlineInput<'a>) {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .write_all(serde_json::to_string(&input).unwrap().as_bytes())
        .unwrap();
}

/// Stops backend server.
pub fn stop(addr: SocketAddr) {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream.write_all("".as_bytes()).unwrap();
}
