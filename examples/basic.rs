extern crate linfa;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
#[macro_use]
extern crate derive_more;

use linfa::{Blueprint, Fit, IncrementalFit, Transformer};
use ndarray::{Array1, ArrayBase, Axis, Data, Ix1, stack};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use std::error::Error;

/// Short-hand notations
type Input<S> = ArrayBase<S, Ix1>;
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
    pub ddof: f64,
    pub mean: f64,
    pub standard_deviation: f64,
}

/// It keeps track of the number of samples seen so far, to allow for
/// incremental computation of mean and standard deviation.
pub struct OnlineOptimizer {
    pub n_samples: u64,
}

/// Initialize n_samples to 0.
impl Default for OnlineOptimizer {
    fn default() -> Self {
        Self { n_samples: 0 }
    }
}

impl<S> Fit<Config, Input<S>, Output> for OnlineOptimizer
where
    S: Data<Elem = f64>,
{
    type Error = ScalingError;

    fn fit(
        &mut self,
        inputs: &Input<S>,
        _targets: &Output,
        blueprint: Config,
    ) -> Result<StandardScaler, Self::Error> {
        if inputs.len() == 0 {
            return Err(ScalingError {});
        }
        // Compute relevant quantities
        let mean = inputs.mean_axis(Axis(0)).into_scalar();
        let standard_deviation = inputs.std_axis(Axis(0), blueprint.ddof).into_scalar();
        // Initialize n_samples using the array length
        self.n_samples = inputs.len() as u64;
        // Return new, tuned scaler
        Ok(StandardScaler {
            ddof: blueprint.ddof,
            mean,
            standard_deviation,
        })
    }
}

impl<S> IncrementalFit<StandardScaler, Input<S>, Output> for OnlineOptimizer
where
    S: Data<Elem = f64>,
{
    type Error = ScalingError;

    fn incremental_fit(
        &mut self,
        inputs: &Input<S>,
        _targets: &Output,
        transformer: StandardScaler,
    ) -> Result<StandardScaler, Self::Error> {
        if inputs.len() == 0 {
            // Nothing to be done
            return Ok(transformer);
        }
        // Compute relevant quantities for the new batch
        let batch_n_samples = inputs.len();
        let batch_mean = inputs.mean_axis(Axis(0)).into_scalar();
        let batch_std = inputs.std_axis(Axis(0), transformer.ddof).into_scalar();

        // Update
        let mean_delta = batch_mean - transformer.mean;
        let new_n_samples = self.n_samples + (batch_n_samples as u64);
        let new_mean =
            transformer.mean + mean_delta * (batch_n_samples as f64) / (new_n_samples as f64);
        let new_std = transformer.standard_deviation
            + batch_std
            + mean_delta.powi(2) * (self.n_samples as f64) * (batch_n_samples as f64)
                / (new_n_samples as f64);

        // Update n_samples
        self.n_samples = new_n_samples;

        // Return tuned scaler
        Ok(StandardScaler {
            ddof: transformer.ddof,
            mean: new_mean,
            standard_deviation: new_std,
        })
    }
}

pub struct Config {
    // Delta degrees of freedom.
    // With ddof = 1, you get the sample standard deviation
    // With ddof = 0, you get the population standard deviation
    ddof: f64,
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

fn generate_batch(n_samples: usize) -> (Array1<f64>, Array1<f64>) {
    let distribution = Uniform::new(0., 10.);
    let x = Array1::random(n_samples, distribution);
    let y = Array1::random(n_samples, distribution);
    (x, y)
}

fn check<S>(scaler: &StandardScaler, x: &ArrayBase<S, Ix1>) -> Result<(), ScalingError>
where
    S: Data<Elem=f64>
{
    let old_batch_mean = x.mean_axis(Axis(0)).into_scalar();
    let new_batch_mean = scaler
        .transform(&x)?
        .mean_axis(Axis(0))
        .into_scalar();
    println!(
        "The mean.\nBefore scaling: {:?}\nAfter scaling: {:?}\n",
        old_batch_mean, new_batch_mean
    );
    Ok(())
}

fn main() -> Result<(), ScalingError> {
    let n_samples = 20;
    let (x, y) = generate_batch(n_samples);

    let mut optimizer = OnlineOptimizer::default();
    let standard_scaler = optimizer.fit(&x, &y, Config::default())?;

    check(&standard_scaler, &x)?;

    let (x2, y2) = generate_batch(n_samples);
    let standard_scaler = optimizer.incremental_fit(&x2, &y2, standard_scaler)?;

    let whole_x = stack(Axis(0), &[x.view(), x2.view()]).expect("Failed to stack arrays");
    check(&standard_scaler, &whole_x)?;

    Ok(())
}
