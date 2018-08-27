//! Mutation helper module.
//! Here we define the structural mutation available and their useful methods.

use rand::{
    distributions::{Distribution, Standard, Weighted, WeightedChoice},
    Rng,
};

/// This enum defines all implemented mutation.
#[derive(Debug, Clone)]
pub enum StructuralMutation {
    // Add a randomly generated sub-network (or sub-genome) to network (or genome).
    SubNetworkAddition,
    // Add a forward or recurrent jumper connection between two neurons in a linear genome.
    JumperAddition,
    // Remove a connection between two neurons in a linear genome.
    ConnectionRemoval,
}


impl Distribution<StructuralMutation> for Standard {
    /// Each StructuralMutation has an associated weight that influences how likely it is to be chosen: higher
    /// weight is more likely.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> StructuralMutation {
        let mut weighted_available_structural_mutation = vec![
            Weighted {
                weight: 1,
                item: StructuralMutation::SubNetworkAddition,
            },
            Weighted {
                weight: 1,
                item: StructuralMutation::JumperAddition,
            },
            Weighted {
                weight: 1,
                item: StructuralMutation::ConnectionRemoval,
            },
        ];
        let weighted_choice = WeightedChoice::new(&mut weighted_available_structural_mutation);

        weighted_choice.sample(rng)
    }
}
