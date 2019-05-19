extern crate linfa;
extern crate ndarray;
#[macro_use]
extern crate derive_more;

use std::error::Error;
use linfa::Transformer;
use ndarray::{ArrayBase, Ix1, Data, Array1};

#[derive(Debug, Eq, PartialEq, From, Display)]
pub struct ScalingError {}

impl Error for ScalingError {}

pub struct StandardScaler {
    mean: f64,
    standard_deviation: f64,
}

impl<S> Transformer<ArrayBase<S, Ix1>, Array1<f64>> for StandardScaler
where
    S: Data<Elem=f64>,
{
    type Error = ScalingError;

    fn transform(&self, inputs: &ArrayBase<S, Ix1>) -> Result<Array1<f64>, Self::Error>
    where
        S: Data<Elem=f64>,
    {
        Ok((inputs - self.mean) / self.standard_deviation)
    }
}

fn main() {
    println!("Hello world!");
}