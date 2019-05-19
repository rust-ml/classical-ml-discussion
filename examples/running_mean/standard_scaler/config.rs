use crate::standard_scaler::{Input, Output, StandardScaler};
use linfa::Blueprint;
use ndarray::Data;

pub struct Config {
    // Delta degrees of freedom.
    // With ddof = 1, you get the sample standard deviation
    // With ddof = 0, you get the population standard deviation
    pub ddof: f64,
}

/// Defaults to computing the sample standard deviation.
impl Default for Config {
    fn default() -> Self {
        Self { ddof: 1. }
    }
}

impl<S> Blueprint<Input<S>, Output> for Config
    where
        S: Data<Elem = f64>,
{
    type Transformer = StandardScaler;
}

