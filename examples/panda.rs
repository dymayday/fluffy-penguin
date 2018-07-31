//! This example will actually be used during the dev process.

extern crate fluffy_penguin;

use fluffy_penguin::cge::network::Network;
// use fluffy_penguin::cge::node::Allele;

fn main() {
    // for i in 0..5 {
    //     let i: f32 = i as f32 - 2.5_f32;
    //     // println!("{:4} : ReLu = {} - SoftPlus = {}", i, activation::relu(i), activation::softplus(i));
    //
    //     println!("{:4} :", i);
    //     println!("     ReLu = {}", activation::relu(i));
    //     // println!("ISRLU 1.0 = {}", activation::isrlu(i, 1_f32));
    //     println!("ISRLU 3.0 = {}", activation::isrlu(i, 3_f32));
    //     println!("ISRLU f64 3.0 = {}", (i as f64).isrlu(3_f64));
    //     // println!(" SoftPlus = {}", activation::softplus(i));
    //     // println!(" ISRU 1.0 = {}", activation::isru(i, 1_f32));
    //     // println!(" ISRU 3.0 = {}", activation::isru(i, 3_f32));
    //     // println!(" Sigmoids = {}", activation::sigmoids(i));
    //     println!("");
    // }
    //
    //
    // let n = cge::node::Neuron::new(0 as usize, 0.3_f32, 1_i32);
    // println!("Neuron = {:#?}", n);


    let mut network: Network<f32> = Network::build_from_example();
    println!("Example Node network :");
    println!("Evaluated output = {:?}", network.evaluate());

}
