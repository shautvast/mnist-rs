use mnist_rs::dataloader::load_data;

fn main() {
    let mut net = mnist_rs::net::Network::gaussian(vec![784, 30, 10]);
    let (training_data, test_data) = load_data();

    net.sgd(training_data, 30, 1, 0.01, Some(test_data));


    // let sizes = vec![5,3,2];
    // let net = mnist_rs::net::Network::from(sizes);
    // println!("biases {:?}", net.biases.iter().map(|b|b.shape()).collect::<Vec<(usize,usize)>>());
    // println!("weights {:?}", net.weights.iter().map(|b|b.shape()).collect::<Vec<(usize,usize)>>());


}