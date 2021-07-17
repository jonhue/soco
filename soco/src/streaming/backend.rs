use crate::algorithms::online::Memory;
use crate::algorithms::online::OnlineAlgorithm;
use crate::algorithms::Options;
use crate::model::{Model, OfflineInput, OnlineInput};
use crate::problem::{Online, Problem};
use crate::schedule::Schedule;
use crate::value::Value;
use std::error::Error;
use std::io::Write;
use std::net::{SocketAddr, TcpListener, TcpStream};

pub fn start<'a, T, P, M, O, A, B>(
    addr: SocketAddr,
    model: &'a impl Model<'a, P, A, B>,
    mut o: Online<P>,
    alg: &impl OnlineAlgorithm<'a, T, P, M, O>,
    xs: &mut Schedule<T>,
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
        match read_request(&mut stream) {
            Ok(input) => {
                println!("Received: {:?}", input);
                model.update(&mut o, input);
                let result = o.next(alg, options.clone(), xs, prev_m).unwrap();
                let response = serde_json::to_string(&result).unwrap();
                stream.write_all(response.as_bytes()).unwrap();
                stream.flush().unwrap();
                prev_m = result.1;
            }
            Err(_) => {
                println!("Stopping server.");
                break;
            }
        }
    }
}

fn read_request<'a, B>(tcp_stream: &mut TcpStream) -> Result<B, Box<dyn Error>>
where
    B: OnlineInput<'a>,
{
    let mut deserializer = serde_json::Deserializer::from_reader(tcp_stream);
    let next = B::deserialize(&mut deserializer)?;
    Ok(next)
}
