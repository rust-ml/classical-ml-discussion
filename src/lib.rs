use std::error::Error;

pub trait Trainer<M>
    where M: Model
{
    type Input;
    type Target;

    fn train(self, inputs: &Self::Input) -> Result<M, Error>;
}

pub trait Model {
    type Input;
    type Output;

    fn predict(&self, inputs: &Self::Input) -> Result<Self::Output, Error>;
}