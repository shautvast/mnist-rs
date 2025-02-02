use mnist_rs::dataloader::load_data;
use std::time::Instant;

fn main() {
    let mut net = mnist_rs::net::Network::gaussian(vec![784, 30, 10]);
    let (training_data, test_data) = load_data();

    let t0 = Instant::now();

    net.sgd(training_data, 30, 10, 3.0, Some(test_data));
    println!("{}", t0.elapsed().as_millis());
}