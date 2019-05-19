use crate::standard_scaler::{Input, Output, ScalingError};
use linfa::Transformer;
use ndarray::Data;

/// Given an input, it rescales it to have zero mean and unit variance.
///
/// We use 64-bit floats for simplicity.
pub struct StandardScaler {
    // Delta degrees of freedom.
    // With ddof = 1, you get the sample standard deviation
    // With ddof = 0, you get the population standard deviation
    pub ddof: f64,
    pub mean: f64,
    pub standard_deviation: f64,
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
