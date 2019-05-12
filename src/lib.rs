use std::error;
use std::iter;

/// One step closer to the peak.
///
/// `Optimizer` is generic over a type `M` implementing the `Model` trait: `M` is used to
/// constrain what type of inputs and targets are acceptable.
///
/// `train` takes an instance of `M` as one of its inputs, `model`: it doesn't matter if `model`
/// has been through several rounds of training before, or if it just came out of a `Blueprint`
/// using `initialize` - it's consumed by `train` and a new model is returned.
///
/// This means that there is no difference between one-shot training and incremental training.
/// Furthermore, the optimizer doesn't have to "own" the model or know anything about its hyperparameters,
/// because it never has to initialize it.
pub trait Optimizer<M>
where
    M: Model,
{
    type Error: error::Error;

    fn train(
        &self,
        inputs: &M::Input,
        targets: &M::Output,
        model: M,
    ) -> Result<M, Self::Error>;
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

/// Any `Blueprint` can be used as `BlueprintGenerator`, as long as it's clonable:
/// it returns an iterator with a single element, a clone of itself.
impl<B, M> BlueprintGenerator<B, M> for B
where
    B: Blueprint<M> + Clone,
    M: Model,
{
    type Error = B::Error;
    type Output = iter::Once<B>;

    fn generate(&self) -> Result<Self::Output, Self::Error>
    {
        Ok(iter::once(self.clone()))
    }
}


/// Where you need to go meta (hyperparameters!).
///
/// `BlueprintGenerator`s can be used to explore different combination of hyperparameters
/// when you are working with a certain `Model` type.
///
/// `BlueprintGenerator::generate` returns, if successful, an `IntoIterator` type
/// yielding instances of blueprints.
pub trait BlueprintGenerator<B, M>
where
    B: Blueprint<M>,
    M: Model
{
    type Error: error::Error;
    type Output: IntoIterator<Item=B>;

    fn generate(&self) -> Result<Self::Output, Self::Error>;
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
