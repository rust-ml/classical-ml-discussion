use std::error;

pub trait Trainer<M>
    where M: Model
{
    type Input;
    type Target;
    type Error: error::Error;

    fn train(self, inputs: &Self::Input) -> Result<M, Self::Error>;
}

/// The basic `Model` trait.
///
/// It is training-agnostic: a model takes an input and returns an output.
///
/// There might be multiple ways to discover the best settings for every
/// particular algorithm (e.g. training a logistic regressor using
/// a pseudo-inverse matrix vs using gradient descent).
/// It doesn't matter: the end result, the model, is a set of parameters.
/// The way those parameter originated is an orthogonal concept.
///
/// In the same way, it has no notion of loss or "correct" predictions.
/// Those concepts are embedded elsewhere.
pub trait Model {
    type Input;
    type Output;
    type Error: error::Error;

    fn predict(&self, inputs: &Self::Input) -> Result<Self::Output, Self::Error>;
}