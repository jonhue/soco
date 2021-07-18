use crate::{
    algorithms::{
        online::{Memory, OnlineAlgorithm},
        Options,
    },
    config::Config,
    model::{Model, OfflineInput, OnlineInput},
    problem::{Online, Problem},
    result::Result,
    schedule::Schedule,
    value::Value,
};
use std::{
    io::Write,
    net::{SocketAddr, TcpListener, TcpStream},
};

/// Generates problem instance from model and streams online algorithm using the provided input.
/// Returns problem instance, initial schedule, and latest memory of the algorithm.
///
/// The returned values can be used to start the backend and stream the algorithm live.
#[allow(clippy::type_complexity)]
pub fn prepare<'a, T, P, M, O, A, B>(
    model: &'a impl Model<'a, P, A, B>,
    alg: &impl OnlineAlgorithm<'a, T, P, M, O>,
    options: O,
    w: i32,
    input: A,
) -> Result<(Online<P>, (Schedule<T>, Option<M>))>
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

    println!("Simulating until time slot {}.", t_end);
    let result = if t_end >= 1 {
        o.offline_stream(&alg, t_end, options)?
    } else {
        (Schedule::empty(), None)
    };
    Ok((o, result))
}

/// Starts backend server.
pub fn start<'a, T, P, M, O, A, B>(
    addr: SocketAddr,
    model: &'a impl Model<'a, P, A, B>,
    mut o: Online<P>,
    alg: &impl OnlineAlgorithm<'a, T, P, M, O>,
    mut xs: Schedule<T>,
    mut prev_m: Option<M>,
    options: O,
) where
    T: Value<'a>,
    P: Problem + 'a,
    M: Memory<'a, P>,
    O: Options<P>,
    A: OfflineInput,
    B: OnlineInput<'a>,
{
    let listener = TcpListener::bind(addr).unwrap();
    println!("Backend is running.");

    for stream in listener.incoming() {
        let mut stream = stream.unwrap();
        println!("Connection established!");
        match serde_json::from_reader(&mut stream) {
            Ok(input) => {
                println!("Received: {:?}", input);
                model.update(&mut o, input);

                let (x, m) =
                    o.next(alg, options.clone(), &mut xs, prev_m).unwrap();
                let response = serde_json::to_string(&x).unwrap();

                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();

                prev_m = m;
            }
            Err(_) => {
                println!("Stopping server.");
                break;
            }
        }
    }
}

/// Stops backend server.
pub fn stop(addr: SocketAddr) {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream.write_all("".as_bytes()).unwrap();
    stream.flush().unwrap();
}

/// Executes next iteration of online algorithm.
pub fn next<'a, T, P, M>(
    addr: SocketAddr,
    input: impl OnlineInput<'a>,
) -> (Config<T>, Option<M>)
where
    T: Value<'a>,
    P: Problem + 'a,
    M: Memory<'a, P>,
{
    let mut stream = TcpStream::connect(addr).unwrap();
    stream
        .write_all(serde_json::to_string(&input).unwrap().as_bytes())
        .unwrap();
    stream.flush().unwrap();
    serde_json::from_reader(&mut stream).unwrap()
}
