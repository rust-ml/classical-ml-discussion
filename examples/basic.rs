extern crate linfa;
extern crate ndarray;
#[macro_use]
extern crate derive_more;

use linfa::{Transformer, Blueprint};
use ndarray::{Array1, ArrayBase, Data, Ix1};
use std::error::Error;

/// Fast-and-dirty error struct
#[derive(Debug, Eq, PartialEq, From, Display)]
pub struct ScalingError {}

impl Error for ScalingError {}

/// Given an input, it rescales it to have zero mean and unit variance.
///
/// We use 64-bit floats for simplicity.
pub struct StandardScaler {
    // Delta degrees of freedom.
    // With ddof = 1, you get the sample standard deviation
    // With ddof = 0, you get the population standard deviation
    ddof: u8,
    mean: f64,
    standard_deviation: f64,
}

/// It keeps track of the number of samples seen so far, to allow for
/// incremental computation of mean and standard deviation.
pub struct OnlineOptimizer {
    n_samples: u64
}

pub struct Config {
    // Delta degrees of freedom.
    // With ddof = 1, you get the sample standard deviation
    // With ddof = 0, you get the population standard deviation
    ddof: u8
}

/// Defaults to computing the sample standard deviation.
impl Default for Config {
    fn default() -> Self {
        Self { ddof: 1 }
    }
}

impl<S> Transformer<ArrayBase<S, Ix1>, Array1<f64>> for StandardScaler
where
    S: Data<Elem = f64>,
{
    type Error = ScalingError;

    fn transform(&self, inputs: &ArrayBase<S, Ix1>) -> Result<Array1<f64>, Self::Error>
    where
        S: Data<Elem = f64>,
    {
        Ok((inputs - self.mean) / self.standard_deviation)
    }
}

fn main() {
    println!("Hello world!");
}
