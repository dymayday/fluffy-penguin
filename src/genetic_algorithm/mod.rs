pub mod individual;
mod mutation;
mod population;

pub use self::individual::{Specimen, LEARNING_RATE_THRESHOLD};
pub use self::population::Population;
