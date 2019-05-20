use ndarray::{Array1, ArrayBase, Ix1};

/// Short-hand notations
type Input<S> = ArrayBase<S, Ix1>;
type Output = Array1<f64>;

mod config;
mod error;
mod optimizer;
mod transformer;

pub use config::Config;
pub use error::ScalingError;
pub use optimizer::OnlineOptimizer;
pub use transformer::StandardScaler;
