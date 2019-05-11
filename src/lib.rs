use std::error;

/// Where information is distilled from data.
///
/// `Optimizer` is generic over a type `M` implementing the `Model` trait: `M` is used to
/// constrain what type of inputs and targets are acceptable, as well as what signature the
/// loss function should have.
///
/// `train` takes an instance of `M` as one of its inputs, `model`: it doesn't matter if `model`
/// has been through several rounds of training before, or if it just came out of a `Blueprint`
/// using `initialize` - it's consumed by `train` and a new model is returned.
///
/// This means that there is no difference between one-shot training and incremental training.
/// Furthermore, the optimizer doesn't have to "own" the model or know anything about its hyperparameters,
/// because it never has to initialize it.
///
/// The output of the loss function is currently unconstrained: should it be an associated
/// type of the `Optimizer` trait itself? Should we add it as a generic parameter of the
/// `train` method, with a set of reasonable trait bounds?
pub trait Optimizer<M>
where
    M: Model,
{
    type Error: error::Error;

    fn train<L>(&self, inputs: &M::Input, targets: &M::Output, model: M, loss: L) -> Result<M, Self::Error>
    // Returning f64 is arbitrary, but I didn't want to flesh out a Loss trait yet
    where
        L: FnMut(&M::Output, &M::Output) -> f64;
}

/// Where `Model`s are forged.
///
/// `Blueprint`s are used to specify how to build and initialize an instance of the model type `M`.
///
/// For the same model type `M`, nothing prevents a user from providing more than one `Blueprint`:
/// multiple initialization strategies can somethings be used to be build the same model type.
///
/// Each of these strategies can take different (hyper)parameters, even though they return an
/// instance of the same model type in the end.
///
/// The initialization procedure could be data-dependent, hence the signature of `initialize`.
pub trait Blueprint<M>
where
    M: Model,
{
    type Error: error::Error;

    fn initialize(&self, inputs: &M::Input, targets: &M::Output) -> Result<M, Self::Error>;
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
