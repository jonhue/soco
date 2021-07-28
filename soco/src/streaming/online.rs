use crate::{
    algorithms::{
        online::{Memory, OnlineAlgorithm},
        Options,
    },
    config::Config,
    convert::{CastableSchedule, DiscretizableSchedule},
    cost::Cost,
    model::{
        Model, ModelOutputFailure, ModelOutputSuccess, OfflineInput,
        OnlineInput,
    },
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

#[derive(Clone)]
pub struct OfflineResponse<T, C, D, M> {
    pub xs: (Schedule<T>, Cost<C, D>),
    pub int_xs: (Schedule<i32>, Cost<C, D>),
    pub m: Option<M>,
}

type OnlineResponse<T, C, D, M> = std::result::Result<
    (
        (Config<T>, Cost<C, D>),
        (Config<i32>, Cost<C, D>),
        Option<M>,
    ),
    String,
>;

/// Generates problem instance from model and streams online algorithm using the provided input.
/// Then, starts backend.
/// Returns initial schedule, initial integral schedule, and latest memory of the algorithm.
pub fn start<T, P, M, O, A, B, C, D>(
    addr: SocketAddr,
    model: impl Model<T, P, A, B, C, D> + 'static,
    alg: &'static impl OnlineAlgorithm<'static, T, P, M, O, C, D>,
    options: O,
    w: i32,
    input: A,
    sender: Option<Sender<&'static str>>,
) -> Result<OfflineResponse<T, C, D, M>>
where
    T: Value<'static>,
    P: Problem<T, C, D> + 'static,
    M: Memory<'static, T, P, C, D>,
    O: Options<T, P, C, D> + 'static,
    A: OfflineInput,
    B: OnlineInput,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
{
    let (mut o, result) = prepare(&model, alg, options.clone(), w, input)?;
    let OfflineResponse {
        xs: (mut xs, _),
        m: prev_m,
        ..
    } = result.clone();

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
fn prepare<'a, T, P, M, O, A, B, C, D>(
    model: &impl Model<T, P, A, B, C, D>,
    alg: &impl OnlineAlgorithm<'a, T, P, M, O, C, D>,
    options: O,
    w: i32,
    input: A,
) -> Result<(Online<P>, OfflineResponse<T, C, D, M>)>
where
    T: Value<'a>,
    P: Problem<T, C, D> + 'a,
    M: Memory<'a, T, P, C, D>,
    O: Options<T, P, C, D> + 'a,
    A: OfflineInput,
    B: OnlineInput,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
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
    let cost = o.p.objective_function(&xs)?;
    let int_xs = xs.to_i();
    let int_cost = o.p.objective_function(&int_xs.to())?;
    Ok((
        o,
        OfflineResponse {
            xs: (xs, cost),
            int_xs: (int_xs, int_cost),
            m,
        },
    ))
}

/// Starts backend server.
#[allow(clippy::too_many_arguments)]
fn run<'a, T, P, M, O, A, B, C, D>(
    addr: SocketAddr,
    model: impl Model<T, P, A, B, C, D>,
    mut o: &mut Online<P>,
    alg: &impl OnlineAlgorithm<'a, T, P, M, O, C, D>,
    mut xs: &mut Schedule<T>,
    mut prev_m: Option<M>,
    options: O,
    sender: Option<Sender<&str>>,
) where
    T: Value<'a>,
    P: Problem<T, C, D> + 'a,
    M: Memory<'a, T, P, C, D>,
    O: Options<T, P, C, D>,
    A: OfflineInput,
    B: OnlineInput,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
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

                    let cost = o.p.objective_function(&xs).unwrap();
                    let int_xs = xs.to_i();
                    let int_cost =
                        o.p.objective_function(&int_xs.to()).unwrap();

                    let result: OnlineResponse<T, C, D, M> =
                        Ok(((x, cost), (int_xs.now(), int_cost), m.clone()));
                    let response = bincode::serialize(&result).unwrap();

                    stream.write_all(&response).unwrap();
                    stream.flush().unwrap();
                    info!("[server] Sent: {:?}", result);

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
                        let result: OnlineResponse<T, C, D, M> =
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
pub fn next<'a, T, P, M, B, C, D>(
    addr: SocketAddr,
    input: B,
) -> OnlineResponse<T, C, D, M>
where
    T: Value<'a>,
    P: Problem<T, C, D> + 'a,
    M: Memory<'a, T, P, C, D>,
    B: OnlineInput,
    C: ModelOutputSuccess,
    D: ModelOutputFailure,
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
