//! Mutation helper module.
//! Here we define the structural mutation available and their useful methods.

use rand::{
    distributions::{Distribution, Standard},
    Rng,
};

/// This enum defines all implemented mutation.
#[derive(Debug)]
pub enum StructuralMutation {
    // Add a randomly generated sub-network (or sub-genome) to network (or genome).
    SubNetworkAddition,
    // Add a forward or recurrent jumper connection between two neurons in a linear genome.
    JumperAddition,
    // Remove a connection between two neurons in a linear genome.
    ConnectionRemoval,
}


impl Distribution<StructuralMutation> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> StructuralMutation {
        match rng.gen_range(0, 3) {
            0 => StructuralMutation::SubNetworkAddition,
            1 => StructuralMutation::JumperAddition,
            _ => StructuralMutation::ConnectionRemoval,
        }
    }
}
