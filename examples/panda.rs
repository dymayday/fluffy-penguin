//! This example will actually be used during the dev process.

extern crate rand;
extern crate fluffy_penguin;

use rand::{thread_rng, Rng};
use fluffy_penguin::cge::network::Network;
// use fluffy_penguin::cge::node::Allele;



/// Dev purpose: this function test the different implementation of the available variation operators:
/// * Structural Mutation
/// * parametric Mutation
fn dev_variation_operator() {
    let rnd_vec: Vec<f32> = vec![1_f32; 16].iter().map(|_| thread_rng().gen_range(-999_f32, 100_f32)).collect();
    println!("Random input vector = {:?}", rnd_vec);
    let mut panda_net: Network<f32> = Network::new_simple(&rnd_vec, 9);
    println!("Evaluated panda_net output = {:?}", panda_net.evaluate());
}

fn main() {
    let mut network: Network<f32> = Network::build_from_example();
    println!("Evaluated example output = {:?}", network.evaluate());

    dev_variation_operator();
}
