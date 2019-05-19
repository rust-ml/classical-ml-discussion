extern crate linfa;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
#[macro_use]
extern crate derive_more;

use crate::standard_scaler::{Config, OnlineOptimizer, ScalingError, StandardScaler};
use linfa::{Fit, IncrementalFit, Transformer};
use ndarray::{stack, Array1, ArrayBase, Axis, Data, Ix1};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

mod standard_scaler;

fn generate_batch(n_samples: usize) -> (Array1<f64>, Array1<f64>) {
    let distribution = Uniform::new(0., 10.);
    let x = Array1::random(n_samples, distribution);
    let y = Array1::random(n_samples, distribution);
    (x, y)
}

fn check<S>(scaler: &StandardScaler, x: &ArrayBase<S, Ix1>) -> Result<(), ScalingError>
where
    S: Data<Elem = f64>,
{
    let old_batch_mean = x.mean_axis(Axis(0)).into_scalar();
    let new_batch_mean = scaler.transform(&x)?.mean_axis(Axis(0)).into_scalar();
    println!(
        "The mean.\nBefore scaling: {:?}\nAfter scaling: {:?}\n",
        old_batch_mean, new_batch_mean
    );
    Ok(())
}

/// Run it with: cargo run --example running_mean
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
