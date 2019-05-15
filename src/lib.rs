use std::error;
use std::iter;

/// The basic `Transformer` trait.
///
/// It is training-agnostic: a transformer takes an input and returns an output.
///
/// There might be multiple ways to discover the best settings for every
/// particular algorithm (e.g. training a logistic regressor using
/// a pseudo-inverse matrix vs using gradient descent).
/// It doesn't matter: the end result, the transformer, is a set of parameters.
/// The way those parameter originated is an orthogonal concept.
///
/// In the same way, it has no notion of loss or "correct" predictions.
/// Those concepts are embedded elsewhere.
pub trait Transformer<I, O> {
    type Error: error::Error;

    fn transform(&self, inputs: &I) -> Result<O, Self::Error>;
}

/// One step closer to the peak.
///
/// `Fit` is generic over a type `B` implementing the `Blueprint` trait: `B::Transformer` is used to
/// constrain what type of inputs and targets are acceptable.
///
/// `fit` takes an instance of `B` as one of its inputs, `blueprint`: it's consumed with move
/// semantics and a new transformer is returned.
///
/// It's a transition in the transformer state machine: from `Blueprint` to `Transformer`.
pub trait Fit<B, I, O>
where
    B: Blueprint<I, O>,
{
    type Error: error::Error;

    fn fit(
        &self,
        inputs: &I,
        targets: &O,
        blueprint: B,
    ) -> Result<B::Transformer, Self::Error>;
}

/// We are not done with that `Transformer` yet.
///
/// `IncrementalFit` is generic over a type `T` implementing the `Transformer` trait: `T` is used to
/// constrain what type of inputs and targets are acceptable.
///
/// `incremental_fit` takes an instance of `T` as one of its inputs, `transformer`: it's consumed with move
/// semantics and a new transformer is returned.
///
/// It's a transition in the transformer state machine: from `Transformer` to `Transformer`.
pub trait IncrementalFit<T, I, O>
where
    T: Transformer<I, O>
{
    type Error: error::Error;

    fn incremental_fit(
        &self,
        inputs: &I,
        targets: &O,
        transformer: T,
    ) -> Result<T, Self::Error>;

}

/// Where `Transformer`s are forged.
///
/// `Blueprint` is a marker trait: it identifies what types can be used as starting points for
/// building `Transformer`s. It's the initial stage of the transformer state machine.
///
/// Every `Blueprint` is associated to a single `Transformer` type (is it wise to do so?).
///
/// For the same transformer type `T`, nothing prevents a user from providing more than one `Blueprint`:
/// multiple initialization strategies can sometimes be used to be build the same transformer type.
///
/// Each of these strategies can take different (hyper)parameters, even though they return an
/// instance of the same transformer type in the end.
pub trait Blueprint<I, O> {
    type Transformer: Transformer<I, O>;
}

/// Where you need to go meta (hyperparameters!).
///
/// `BlueprintGenerator`s can be used to explore different combination of hyperparameters
/// when you are working with a certain `Transformer` type.
///
/// `BlueprintGenerator::generate` returns, if successful, an `IntoIterator` type
/// yielding instances of blueprints.
pub trait BlueprintGenerator<B, I, O>
where
    B: Blueprint<I, O>,
{
    type Error: error::Error;
    type Output: IntoIterator<Item=B>;

    fn generate(&self) -> Result<Self::Output, Self::Error>;
}

/// Any `Blueprint` can be used as `BlueprintGenerator`, as long as it's clonable:
/// it returns an iterator with a single element, a clone of itself.
impl<B, I, O> BlueprintGenerator<B, I, O> for B
    where
        B: Blueprint<I, O> + Clone,
{
    // Random error, didn't have time to get a proper one
    type Error = std::io::Error;
    type Output = iter::Once<B>;

    fn generate(&self) -> Result<Self::Output, Self::Error>
    {
        Ok(iter::once(self.clone()))
    }
}
