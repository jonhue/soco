use crate::{
    algorithms::{
        online::{Memory, OnlineAlgorithm},
        Options,
    },
    config::Config,
    convert::{CastableSchedule, DiscretizableSchedule},
    model::{Model, OfflineInput, OnlineInput},
    objective::Objective,
    problem::{Online, Problem},
    result::Result,
    schedule::Schedule,
    value::Value,
};
use backtrace::Backtrace;
use log::{info, warn};
use std::{
    io::Write,
    net::{SocketAddr, TcpListener, TcpStream},
    panic,
    sync::{mpsc::Sender, Mutex},
    thread,
};

type OnlineResponse<T, M> = std::result::Result<
    ((Config<T>, f64), (Config<i32>, f64), Option<M>),
    String,
>;

/// Generates problem instance from model and streams online algorithm using the provided input.
/// Then, starts backend.
/// Returns initial schedule, initial integral schedule, and latest memory of the algorithm.
#[allow(clippy::type_complexity)]
pub fn start<T, P, M, O, A, B>(
    addr: SocketAddr,
    model: impl Model<P, A, B> + 'static,
    alg: &'static impl OnlineAlgorithm<'static, T, P, M, O>,
    options: O,
    w: i32,
    input: A,
    sender: Option<Sender<&'static str>>,
) -> Result<((Schedule<T>, f64), (Schedule<i32>, f64), Option<M>)>
where
    T: Value<'static>,
    P: Objective<'static, T> + Problem + 'static,
    M: Memory<'static, P>,
    O: Options<P> + 'static,
    A: OfflineInput,
    B: OnlineInput,
{
    let (mut o, result) = prepare(&model, alg, options.clone(), w, input)?;
    let ((mut xs, _), _, prev_m) = result.clone();

    thread::spawn(move || {
        run(addr, model, &mut o, alg, &mut xs, prev_m, options, sender);
    });

    Ok(result)
}

/// Generates problem instance from model and streams online algorithm using the provided input.
/// Returns problem instance, initial schedule, initial integral schedule, and latest memory of the algorithm.
///
/// The returned values can be used to start the backend and stream the algorithm live.
#[allow(clippy::type_complexity)]
fn prepare<'a, T, P, M, O, A, B>(
    model: &impl Model<P, A, B>,
    alg: &impl OnlineAlgorithm<'a, T, P, M, O>,
    options: O,
    w: i32,
    input: A,
) -> Result<(
    Online<P>,
    ((Schedule<T>, f64), (Schedule<i32>, f64), Option<M>),
)>
where
    T: Value<'a>,
    P: Objective<'a, T> + Problem + 'a,
    M: Memory<'a, P>,
    O: Options<P> + 'a,
    A: OfflineInput,
    B: OnlineInput,
{
    let mut p = model.to(input);
    let t_end = p.t_end() - w;
    if p.t_end() > 1 {
        p.set_t_end(1);
    }
    let mut o = Online { p, w };
    o.verify()?;
    info!("Generated a problem instance: {:?}", o);

    info!("Simulating until time slot {}.", t_end);
    let (xs, m) = if t_end >= 1 {
        o.offline_stream(&alg, t_end, options)?
    } else {
        (Schedule::empty(), None)
    };
    let cost = o.p.objective_function(&xs)?.raw();
    let int_xs = xs.to_i();
    let int_cost = o.p.objective_function(&int_xs.to())?.raw();
    Ok((o, ((xs, cost), (int_xs, int_cost), m)))
}

/// Starts backend server.
#[allow(clippy::too_many_arguments)]
fn run<'a, T, P, M, O, A, B>(
    addr: SocketAddr,
    model: impl Model<P, A, B>,
    mut o: &mut Online<P>,
    alg: &impl OnlineAlgorithm<'a, T, P, M, O>,
    mut xs: &mut Schedule<T>,
    mut prev_m: Option<M>,
    options: O,
    sender: Option<Sender<&str>>,
) where
    T: Value<'a>,
    P: Objective<'a, T> + Problem + 'a,
    M: Memory<'a, P>,
    O: Options<P>,
    A: OfflineInput,
    B: OnlineInput,
{
    let listener = TcpListener::bind(addr).unwrap();
    info!("[server] Running on {:?}.", addr);
    if let Some(sender) = sender {
        sender.send("[server] Running.").unwrap()
    }

    for stream in listener.incoming() {
        let mut stream = stream.unwrap();
        info!("[server] Connection established!");

        match bincode::deserialize_from(&mut stream) {
            Ok(input) => {
                let stream_ref = Mutex::new(&mut stream);
                let model_ref = Mutex::new(&model);
                let o_ref = Mutex::new(o);
                let alg_ref = Mutex::new(alg);
                let xs_ref = Mutex::new(xs);
                let prev_m_ref = Mutex::new(prev_m);
                let options_ref = Mutex::new(&options);

                match panic::catch_unwind(|| {
                    panic::set_hook(Box::new(|_panic_info| {
                        warn!("\n\n{:?}", Backtrace::new());
                    }));

                    let stream = stream_ref.into_inner().unwrap();
                    let model = model_ref.into_inner().unwrap();
                    let o = o_ref.into_inner().unwrap();
                    let alg = alg_ref.into_inner().unwrap();
                    let xs = xs_ref.into_inner().unwrap();
                    let prev_m = prev_m_ref.into_inner().unwrap();
                    let options = options_ref.into_inner().unwrap();

                    info!("[server] Received: {:?}", input);
                    model.update(o, input);
                    info!("[server] Updated problem instance.");

                    let (x, m) =
                        o.next(alg, options.clone(), xs, prev_m).unwrap();

                    let cost = o.p.objective_function(&xs).unwrap().raw();
                    let int_xs = xs.to_i();
                    let int_cost =
                        o.p.objective_function(&int_xs.to()).unwrap().raw();

                    let result: OnlineResponse<T, M> =
                        Ok(((x, cost), (int_xs.now(), int_cost), m.clone()));
                    let response = bincode::serialize(&result).unwrap();

                    stream.write_all(&response).unwrap();
                    stream.flush().unwrap();
                    info!("[server] Sent result: {:?}", result);

                    (o, xs, m)
                }) {
                    Ok((new_o, new_xs, m)) => {
                        o = new_o;
                        xs = new_xs;
                        prev_m = m
                    }
                    Err(panic_) => {
                        let panic = panic_.downcast::<&str>().unwrap();
                        warn!("[server] ERROR (unrecoverable): {:?}", panic);
                        let result: OnlineResponse<T, M> =
                            Err(panic.to_string());
                        let response = bincode::serialize(&result).unwrap();
                        stream.write_all(&response).unwrap();
                        stream.flush().unwrap();
                        break;
                    }
                };
            }
            Err(_) => {
                info!("[server] Server stopped.");
                break;
            }
        }
    }
}

/// Executes next iteration of online algorithm.
/// Returns obtained result, integral result, and memory.
#[allow(clippy::type_complexity)]
pub fn next<'a, T, P, M, B>(addr: SocketAddr, input: B) -> OnlineResponse<T, M>
where
    T: Value<'a>,
    P: Problem + 'a,
    M: Memory<'a, P>,
    B: OnlineInput,
{
    let mut stream = TcpStream::connect(addr).unwrap();
    info!("[client] Connected to {:?}.", addr);
    stream
        .write_all(&bincode::serialize(&input).unwrap())
        .unwrap();
    stream.flush().unwrap();
    info!("[client] Sent: {:?}", input);
    let result = bincode::deserialize_from(&mut stream).unwrap();
    info!("[client] Received: {:?}", result);
    result
}

/// Stops backend server.
pub fn stop(addr: SocketAddr) {
    let mut stream = TcpStream::connect(addr).unwrap();
    info!("[client] Connected to {:?}.", addr);
    stream.write_all("".as_bytes()).unwrap();
    stream.flush().unwrap();
    info!("[client] Stopping server.");
}
