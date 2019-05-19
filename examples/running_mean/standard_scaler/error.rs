use std::error::Error;

/// Fast-and-dirty error struct
#[derive(Debug, Eq, PartialEq, From, Display)]
pub struct ScalingError {}

impl Error for ScalingError {}
