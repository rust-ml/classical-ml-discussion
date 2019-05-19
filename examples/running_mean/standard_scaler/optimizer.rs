use crate::standard_scaler::{Config, Input, Output, ScalingError, StandardScaler};
use linfa::{Fit, IncrementalFit};
use ndarray::{Axis, Data};

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

        let ddof = transformer.ddof;

        // Compute relevant quantities for the new batch
        let batch_n_samples = inputs.len();
        let batch_mean = inputs.mean_axis(Axis(0)).into_scalar();
        let batch_std = inputs.std_axis(Axis(0), ddof).into_scalar();

        // Update
        let mean_delta = batch_mean - transformer.mean;
        let new_n_samples = self.n_samples + (batch_n_samples as u64);
        let new_mean =
            transformer.mean + mean_delta * (batch_n_samples as f64) / (new_n_samples as f64);
        let new_std = ((transformer.standard_deviation.powi(2) * (self.n_samples as f64 - ddof)
            + batch_std.powi(2) * (batch_n_samples as f64 - ddof)
            + mean_delta.powi(2) * (self.n_samples as f64) * (batch_n_samples as f64)
                / (new_n_samples as f64))
            / (new_n_samples as f64 - ddof))
            .sqrt();

        // Update n_samples
        self.n_samples = new_n_samples;

        // Return tuned scaler
        Ok(StandardScaler {
            ddof,
            mean: new_mean,
            standard_deviation: new_std,
        })
    }
}
