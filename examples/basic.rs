extern crate linfa;
extern crate ndarray;
#[macro_use]
extern crate derive_more;

use linfa::{Blueprint, Fit, IncrementalFit, Transformer};
use ndarray::{Array1, ArrayBase, Data, Ix1};
use std::error::Error;

/// Short-hand notations
type Input<S: Data<Elem = f64>> = ArrayBase<S, Ix1>;
type Output = Array1<f64>;

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
    n_samples: u64,
}

impl<S> Fit<Config, Input<S>, Output> for OnlineOptimizer
where
    S: Data<Elem = f64>,
{
    type Error = ScalingError;

    fn fit(
        &self,
        inputs: &Input<S>,
        targets: &Output,
        blueprint: Config,
    ) -> Result<StandardScaler, Self::Error> {
        unimplemented!()
    }
}

impl<S> IncrementalFit<StandardScaler, Input<S>, Output> for OnlineOptimizer
where
    S: Data<Elem = f64>,
{
    type Error = ScalingError;

    fn incremental_fit(
        &self,
        inputs: &Input<S>,
        targets: &Output,
        transformer: StandardScaler,
    ) -> Result<StandardScaler, Self::Error> {
        unimplemented!()
    }
}

pub struct Config {
    // Delta degrees of freedom.
    // With ddof = 1, you get the sample standard deviation
    // With ddof = 0, you get the population standard deviation
    ddof: u8,
}

/// Defaults to computing the sample standard deviation.
impl Default for Config {
    fn default() -> Self {
        Self { ddof: 1 }
    }
}

impl<S> Blueprint<Input<S>, Output> for Config
where
    S: Data<Elem = f64>,
{
    type Transformer = StandardScaler;
}

impl<S> Transformer<Input<S>, Output> for StandardScaler
where
    S: Data<Elem = f64>,
{
    type Error = ScalingError;

    fn transform(&self, inputs: &Input<S>) -> Result<Output, Self::Error>
    where
        S: Data<Elem = f64>,
    {
        Ok((inputs - self.mean) / self.standard_deviation)
    }
}

fn main() {
    println!("Hello world!");
}
