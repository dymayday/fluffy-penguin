

use rand::distributions::StandardNormal;
use rand::{thread_rng, Rng};
use cge::network::Network;
use cge::node::Node;

pub const LEARNING_RATE_THRESHOLD: f32 = 0.01;

#[derive(Clone, Debug, PartialEq)]
pub struct Specimen<T> {
    input_size: usize,
    output_size: usize,
    // The ANN.
    pub ann: Network<T>,
    // Symbolizes how well an individual solves a problem.
    pub fitness: T,
}

impl Specimen<f32> {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Specimen { 
            input_size,
            output_size,
            ann: Network::<f32>::new(input_size, output_size),
            fitness: 0.0,
        }
    }


    /// Build a Specimen from the example in our research papers.
    /// This corresponds to an ANN with 2 inputs and 1 output.
    pub fn new_from_example() -> Self {
        Specimen {
            input_size: 1_usize,
            output_size: 2_usize,
            ann: Network::<f32>::build_from_example(),
            fitness: 0.0,
        }}


    /// The exploitation phase researches the optimal weight of each Node in the current artificial
    /// neural network.
    pub fn exploitation(&mut self) {
        // Number of chromosome defining the linear genome.
        let n: f32 = self.ann.genome.len() as f32;

        // The proportionality constant.
        let tau: f32 = 1.0 / ( 2.0 * n.sqrt() ).sqrt();
        let tau_p: f32 = 1.0 / ( 2.0 * n ).sqrt();
    
        // Denotes a draw from the standard normal distribution.
        let nu: f32 = thread_rng().sample(StandardNormal) as f32;
        

        for mut node in &mut self.ann.genome {
            // Learning rate value of the current chromosome.
            let sigma: f32 = node.sigma;

            // denotes a separate draw from the standard normal distribution for each node.
            let nu_i: f32 = thread_rng().sample(StandardNormal) as f32;

            // Compute the learning rate matated value.
            let mut sigma_p: f32 = sigma * ( tau_p * nu + tau * nu_i ).exp() as f32;

            // Since standard deviations very close to zero are unwanted (they will have on average
            // a negligible effect), the following boundary rule is used to force step
            // sizes to be no smaller than a pre-defined threshold.
            if sigma_p < LEARNING_RATE_THRESHOLD { sigma_p = LEARNING_RATE_THRESHOLD; }

            // Compute a new mutated connection weight.
            let w_p: f32 = node.w + sigma_p * nu_i;

            // Assign the new mutated learning rate value to the Node.
            node.sigma = sigma_p;
            // Assign the new mutated weight to the Node.
            node.w = w_p;
        }

    }
}
