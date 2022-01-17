//! Mutation helper module.
//! Here we define the structural mutation available and their useful methods.

// use rand::{
//     distributions::{Distribution, Standard, Weighted, WeightedChoice},
//     Rng,
// };
use rand::distributions::{Standard, WeightedIndex};
use rand::prelude::*;

/// This enum defines all implemented mutation.
#[derive(Debug, Clone)]
pub enum StructuralMutation {
    /// Add a randomly generated sub-network (or sub-genome) to network (or genome).
    SubNetworkAddition,
    /// Add a forward or recurrent jumper connection between two neurons in a linear genome.
    JumperAddition,
    /// Remove a connection between two neurons in a linear genome.
    ConnectionRemoval,
}


impl Distribution<StructuralMutation> for Standard {
    /// Each StructuralMutation has an associated weight that influences how likely it is to be chosen: higher
    /// weight is more likely.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> StructuralMutation {
        let weighted_available_structural_mutations = vec![
            (25, StructuralMutation::SubNetworkAddition),
            (100, StructuralMutation::JumperAddition),
            (100, StructuralMutation::ConnectionRemoval),
        ];
        let distrib = WeightedIndex::new(
            weighted_available_structural_mutations
                .iter()
                .map(|item| item.0),
        )
        .expect("Fail to draw a mutation from a weighted choice.");
        weighted_available_structural_mutations[distrib.sample(rng)]
            .1
            .to_owned()
    }
}
