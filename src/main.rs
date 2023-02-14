use mnist_rs::dataloader::load_data;

fn main() {
    let mut net = mnist_rs::net::Network::from(vec![784, 30, 10]);
    for w in net.weights.iter() {
        println!("{}, {}", w.shape().0, w.shape().1);
    }
    println!();
    for b in net.biases.iter() {
        println!("{:?}", b.shape());
    }
    let training_data = load_data();

    net.sgd(training_data, 30, 10, 3.0, &None);
}