use std::iter::zip;
use std::ops::Add;

use nalgebra::DMatrix;
use rand::prelude::*;
use rand_distr::Normal;

use crate::dataloader::{Data, DataLine, OneHotVector};
use crate::mat::add;

#[derive(Debug)]
pub struct Network {
    _sizes: Vec<usize>,
    num_layers: usize,
    pub biases: Vec<DMatrix<f32>>,
    pub weights: Vec<DMatrix<f32>>,
}

impl Network {
    /// The list `sizes` contains the number of neurons in the
    /// respective layers of the network.  For example, if the list
    /// was [2, 3, 1] then it would be a three-layer network, with the
    /// first layer containing 2 neurons, the second layer 3 neurons,
    /// and the third layer 1 neuron. The biases and weights for the
    /// network are initialized randomly, using a Gaussian
    /// distribution with mean 0, and variance 1.  Note that the first
    /// layer is assumed to be an input layer, and by convention we
    /// won't set any biases for those neurons, since biases are only
    /// ever used in computing the outputs from later layers.
    pub fn from(sizes: Vec<usize>) -> Self {
        Self {
            _sizes: sizes.clone(),
            num_layers: sizes.len(),
            biases: biases(sizes[1..].to_vec()),
            weights: weights(zip(sizes[..sizes.len() - 1].to_vec(), sizes[1..].to_vec()).collect()),
        }
    }

    fn feed_forward(&self, input: Vec<f32>) -> Vec<f32> {
        let mut a = DMatrix::from_vec(input.len(), 1, input);
        for (b, w) in zip(&self.biases, &self.weights) {
            a = add(b.clone(), w * a).unwrap();
            a.apply(sigmoid_inplace);
        }
        a.column(1).iter().copied().collect()
    }

    pub fn sgd(&mut self, mut training_data: Data<f32, OneHotVector>, epochs: usize, minibatch_size: usize, eta: f32, test_data: &Option<Data<f32, OneHotVector>>) {
        for j in 0..epochs {
            training_data.shuffle();
            let mini_batches = training_data.as_batches(minibatch_size);
            for mini_batch in mini_batches {
                self.update_mini_batch(mini_batch, eta);
            }

            if let Some(test_data) = test_data {
                println!("Epoch {}: {} / {}", j, self.evaluate(test_data), test_data.len());
            } else {
                println!("Epoch {} complete", j);
            }
        }
    }

    /// Update the network's weights and biases by applying
    /// gradient descent using backpropagation to a single mini batch.
    /// The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
    /// is the learning rate.
    fn update_mini_batch(&mut self, mini_batch: &[DataLine<f32, OneHotVector>], eta: f32) {
        let mut nabla_b: Vec<DMatrix<f32>> = self.biases.iter()
            .map(|b| b.shape())
            .map(|s| DMatrix::zeros(s.0, s.1))
            .collect();
        let mut nabla_w: Vec<DMatrix<f32>> = self.weights.iter()
            .map(|w| w.shape())
            .map(|s| DMatrix::zeros(s.0, s.1))
            .collect();
        for line in mini_batch.iter() {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(line.inputs.to_vec(), &line.label);

            nabla_b = zip(&nabla_b, &delta_nabla_b).map(|(nb, dnb)| nb.add(dnb)).collect();
            nabla_w = zip(&nabla_w, &delta_nabla_w).map(|(nw, dnw)| nw.add(dnw)).collect();
        }

        self.weights = zip(&self.weights, &nabla_w)
            .map(|(w, nw)| (w.add_scalar(-eta / mini_batch.len() as f32)).component_mul(nw)).collect();
        self.biases = zip(&self.biases, &nabla_b)
            .map(|(b, nb)| (b.add_scalar(-eta / mini_batch.len() as f32)).component_mul(nb)).collect();
    }

    /// Return the number of test inputs for which the neural
    /// network outputs the correct result. Note that the neural
    /// network's output is assumed to be the index of whichever
    /// neuron in the final layer has the highest activation.
    fn evaluate(&self, test_data: &Data<f32, OneHotVector>) -> usize {
        let test_results: Vec<(usize, usize)> = test_data.0.iter()
            .map(|line| (argmax(self.feed_forward(line.inputs.clone())), line.label.val))
            .collect();
        test_results.into_iter().filter(|(x, y)| x == y).count()
    }

    /// Return a tuple `(nabla_b, nabla_w)` representing the
    /// gradient for the cost function C_x.  `nabla_b` and
    /// `nabla_w` are layer-by-layer lists of matrices, similar
    /// to `self.biases` and `self.weights`.
    fn backprop(&self, x: Vec<f32>, y: &OneHotVector) -> (Vec<DMatrix<f32>>, Vec<DMatrix<f32>>) {
        // zero_grad ie. set gradient to zero
        let mut nabla_b: Vec<DMatrix<f32>> = self.biases.iter()
            .map(|b| b.shape())
            .map(|s| DMatrix::zeros(s.0, s.1))
            .collect();
        let mut nabla_w: Vec<DMatrix<f32>> = self.weights.iter()
            .map(|w| w.shape())
            .map(|s| DMatrix::zeros(s.0, s.1))
            .collect();

        // feedforward
        let mut activation = DMatrix::from_vec(x.len(), 1, x);
        let mut activations = vec![activation.clone()];
        let mut zs = vec![];

        for (b, w) in zip(&self.biases, &self.weights) {
            let z = add(w * &activation, b.clone()).unwrap();
            zs.push(z.clone());
            activation = z.map(sigmoid);
            activations.push(activation.clone());
        }
        // backward pass
        // delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        let delta: DMatrix<f32> = self.cost_derivative(&activations[activations.len() - 1], y).component_mul(&zs[zs.len() - 1].map(sigmoid_prime));
        let index = nabla_b.len() - 1;
        nabla_b[index] = delta.clone();

        let index = nabla_w.len() - 1;
        let ac = &activations[activations.len() - 2].transpose();
        nabla_w[index] = &delta * ac;
        let lens_zs = zs.len();
        for l in 2..self.num_layers {
            let z = &zs[lens_zs - l];
            let sp = z.map(sigmoid_prime);
            let weight = self.weights[self.weights.len() - l + 1].transpose();
            let delta2 = (weight * &delta).component_mul(&sp);
            let len_nb = nabla_b.len();
            nabla_b[len_nb - l] = delta2.clone();
            let len_nw = nabla_w.len();
            nabla_w[len_nw - l] = delta2 * activations[activations.len() - l - 1].transpose();
        }

        (nabla_b, nabla_w)
    }

    fn cost_derivative(&self, output_activations: &DMatrix<f32>, y: &OneHotVector) -> DMatrix<f32> {
        // output_activations - y
        let shape = output_activations.shape();
        DMatrix::from_iterator(shape.0, shape.1, output_activations.iter().enumerate()
            .map(|(index, a)| a - y.get(index)))
    }
}

fn argmax(val: Vec<f32>) -> usize {
    let mut max = 0.0;
    let mut index = 0;
    for (i, x) in val.iter().enumerate() {
        if *x > max {
            index = i;
            max = *x;
        }
    }
    index
}

fn biases(sizes: Vec<usize>) -> Vec<DMatrix<f32>> {
    sizes.iter().map(|size| random_matrix(*size, 1)).collect()
}

fn weights(sizes: Vec<(usize, usize)>) -> Vec<DMatrix<f32>> {
    sizes.iter().map(|size| random_matrix(size.1, size.0)).collect()
}

fn random_matrix(rows: usize, cols: usize) -> DMatrix<f32> {
    let normal: Normal<f32> = Normal::new(0.0, 1.0).unwrap();

    DMatrix::from_fn(rows, cols, |_, _| normal.sample(&mut thread_rng()))
}

fn sigmoid_inplace(val: &mut f32) {
    *val = sigmoid(*val);
}

fn sigmoid(val: f32) -> f32 {
    1.0 / (1.0 + (-val).exp())
}

/// Derivative of the sigmoid function.
fn sigmoid_prime(val: f32) -> f32 {
    sigmoid(val) * (1.0 - sigmoid(val))
}

#[cfg(test)]
mod test {
    use nalgebra::DMatrix;

    use super::*;

    #[test]
    fn test_sigmoid() {
        let mut mat: DMatrix<f32> = DMatrix::from_vec(1, 1, vec![0.0]);
        mat.apply(sigmoid_inplace);
        assert_eq!(mat.get(0), Some(&0.5));
    }
}