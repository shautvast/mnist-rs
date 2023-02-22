use std::iter::zip;
use nalgebra::DMatrix;

use rand::prelude::*;
use serde::Deserialize;

pub fn load_data() -> Data<f32, OneHotVector> {
    // the mnist data is structured as
    // x: [[[pixels]],[[pixels]], etc],
    // y: [label1, label2, etc]
    // this is transformed to:
    // Data : Vec<DataLine>
    // DataLine {inputs: Vec<pixels as f32>, label: f32}
    let raw_data: RawData = serde_json::from_slice(include_bytes!("data/unittest.json")).unwrap();
    let mut vec = Vec::new();
    for (x, y) in zip(raw_data.x, raw_data.y) {
        vec.push(DataLine { inputs: x, label: onehot(y) });
    }

    Data(vec)
}

#[derive(Deserialize)]
struct RawData {
    x: Vec<Vec<f32>>,
    y: Vec<u8>,
}

/// X is type of input
/// Y is type of output
pub struct DataLine<X, Y> {
    pub inputs: Vec<X>,
    pub label: Y,
}

pub struct Data<X, Y>(pub Vec<DataLine<X, Y>>);

pub struct OneHotVector{
    pub val: usize
}

impl OneHotVector{
    fn new(val: usize) -> Self{
        Self{
            val
        }
    }

    pub fn get(&self, index: usize) -> f32{
        if self.val == index {
            1.0
        } else {
            0.0
        }
    }


}

impl<X, Y> Data<X, Y> {
    pub fn shuffle(&mut self) {
        let mut rng = thread_rng();
        self.0.shuffle(&mut rng);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn as_batches(&self, batch_size: usize) -> Vec<&[DataLine<X, Y>]> {
        let mut batches = Vec::with_capacity(self.0.len() / batch_size + 1);
        let mut offset = 0;
        for _ in 0..self.0.len() / batch_size {
            batches.push(&self.0[offset..offset + batch_size]);
            offset += batch_size;
        }
        batches.push(&self.0[offset..self.0.len()]);
        batches
    }
}

/// returns a vector as matrix where y is one-hot encoded
fn onehot(y: u8) -> OneHotVector {
   OneHotVector::new(y as usize)
}