use std::fmt::Debug;
use nalgebra::DMatrix;

use rand::prelude::*;
use serde::Deserialize;

pub fn load_data() -> (Data<f32, OneHotVector>, Data<f32, OneHotVector>)
{
    // the mnist data is structured as
    // x: [[[pixels]],[[pixels]], etc],
    // y: [label1, label2, etc]
    // this is transformed to:
    // Data : Vec<DataLine>
    // DataLine {inputs: Vec<pixels as f64>, label: f64}
    let raw_training_data: Vec<RawData> =
        serde_json::from_slice(include_bytes!("data/training_data.json")).unwrap();
    let raw_test_data: Vec<RawData> =
        serde_json::from_slice(include_bytes!("data/test_data.json")).unwrap();

    let train = vectorize(raw_training_data);
    let test = vectorize(raw_test_data);

    (Data(train), Data(test))
}

fn vectorize(raw_data: Vec<RawData>) -> Vec<DataLine<f32, OneHotVector>>
{
    let mut result = Vec::new();
    for line in raw_data {
        result.push(DataLine { inputs: DMatrix::from_vec(line.x.len(), 1, line.x), label: onehot(line.y) });
    }
    result
}

#[derive(Deserialize)]
struct RawData
{
    x: Vec<f32>,
    y: u8,
}

/// X is type of input
/// Y is type of output
#[derive(Debug, Clone)]
pub struct DataLine<X, Y> where X: Clone, Y: Clone {
    pub inputs: DMatrix<X>,
    pub label: Y,
}

/// simple way to encode a onehot vector. An object that returns 1.0 if you get the 'right' index, or 0.0 otherwise
#[derive(Debug, Clone, PartialEq)]
pub struct OneHotVector {
    pub val: usize,
}

impl OneHotVector{
    pub fn new(val: usize) -> Self {
        Self { val }
    }

    pub fn get(&self, index: usize) -> f32 {
        if self.val == index {
            1.0
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct Data<X, Y>(pub Vec<DataLine<X, Y>>) where X: Clone, Y: Clone;

impl<X, Y> Data<X, Y> where X: Clone, Y: Clone {
    pub fn shuffle(&mut self) {
        let mut rng = rand::rng();
        self.0.shuffle(&mut rng);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_batches(&self, batch_size: usize) -> Vec<&[DataLine<X, Y>]> {
        let mut batches = Vec::with_capacity(self.0.len() / batch_size + 1);
        let mut offset = 0;
        for _ in 0..self.0.len() / batch_size {
            batches.push(&self.0[offset..offset + batch_size]);
            offset += batch_size;
        }
        batches
    }
}

/// returns a vector as matrix where y is one-hot encoded
fn onehot(y: u8) -> OneHotVector {
    OneHotVector::new(y as usize)
}