//! This example will actually be used during the dev process.

extern crate fluffy_penguin;

use fluffy_penguin::cge::network::Network;
// use fluffy_penguin::cge::node::Allele;

fn main() {
    let mut network: Network<f32> = Network::build_from_example();
    println!("Evaluated example output = {:?}", network.evaluate());

    let mut panda_net: Network<f32> = Network::new_simple(&vec![1_f32; 16], 9);
    println!("Evaluated panda_net output = {:?}", panda_net.evaluate());
}
