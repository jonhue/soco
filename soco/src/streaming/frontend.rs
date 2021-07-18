use std::{io::Write, net::{SocketAddr, TcpStream}};

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
/// Returns initial schedule and latest memory of the algorithm.
pub fn start<'a, T, P, M, O, A, B>(
    addr: SocketAddr,
    model: &'a impl Model<'a, P, A, B>,
    alg: &impl OnlineAlgorithm<'a, T, P, M, O>,
    options: O,
    w: i32,
    input: A,
) -> Result<(Schedule<T>, Option<M>)>
where
    T: Value<'a>,
    P: Problem + 'a,
    M: Memory<'a, P>,
    O: Options<P> + 'a,
    A: OfflineInput,
    B: OnlineInput<'a>,
{
    let mut p = model.to(input);
    let t_end = p.t_end() - w;
    if p.t_end() > 1 {
        p.set_t_end(1);
    }
    let mut o = Online { p, w };
    o.verify()?;
    println!("Generated a problem instance: {:?}", o);

    // stream until current time slot
    println!("Simulating until time slot {}.", t_end);
    let result = if t_end >= 1 {
        o.offline_stream(&alg, t_end, options.clone())?
    } else {
        (Schedule::empty(), None)
    };
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
    stream.flush().unwrap();
}

/// Stops backend server.
pub fn stop(addr: SocketAddr) {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream.write("".as_bytes()).unwrap();
    stream.flush().unwrap();
}
