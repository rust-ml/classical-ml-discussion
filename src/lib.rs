use std::error;

/// Where information is distilled from data.
///
/// `Optimizer` is generic over a type `M` implementing the `Model` trait: `M` is used to
/// constrain what type of inputs and targets are acceptable, as well as what signature the
/// loss function should have.
///
/// The output of the loss function is currently unconstrained: should it be an associated
/// type of the `Optimizer` trait itself? Should we add it as a generic parameter of the
/// `train` method, with a set of reasonable trait bounds?
pub trait Optimizer<M>
where
    M: Model,
{
    type Error: error::Error;

    fn train<L>(&self, inputs: &M::Input, targets: &M::Output, loss: L) -> Result<M, Self::Error>
    // Returning f64 is arbitrary, but I didn't want to flesh out a Loss trait yet
    where
        L: FnMut(&M::Output, &M::Output) -> f64;
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
