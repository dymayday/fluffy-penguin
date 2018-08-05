//! This example will actually be used during the dev process.

extern crate fluffy_penguin;
extern crate rand;

use fluffy_penguin::cge::network::Network;
use rand::{thread_rng, Rng};
// use fluffy_penguin::cge::node::Allele;


/// Dev purpose: this function test the different implementation of the available variation operators:
/// * Structural Mutation
/// * parametric Mutation
fn _dev_variation_operator() {
    let rnd_vec: Vec<f32> = vec![1_f32; 16]
        .iter()
        .map(|_| thread_rng().gen_range(-999_f32, 100_f32))
        .collect();
    println!("Random input vector = {:?}", rnd_vec);
    let mut panda_net: Network<f32> = Network::new_simple(rnd_vec.len(), 9);
    panda_net.update_input(&rnd_vec);

    println!("Evaluated panda_net output = {:?}", panda_net.evaluate());
    println!("");
}


/// Test learning rate and weight mutation.
fn _exploitation() {
    use fluffy_penguin::cge::node::Node;
    use rand::distributions::StandardNormal;

    let n: f64 = 16.0 * 9.0;
    let threshold: f64 = 0.01;
    let mut sigma: f64 = 0.01;
    // let mut sigma: f64 = 1.0;

    let nu: f64 = thread_rng().sample(StandardNormal);
    println!("\nN = {}", nu);

    let mut w: f64 = Node::random_weight() as f64;
    // let mut w: f64 = 0.0;
    println!("w = {}", w);

    let tho: f64 = 1.0 / (2.0 * n.sqrt()).sqrt();
    let tho_p: f64 = 1.0 / (2.0 * n).sqrt();

    println!(
        "{:<3} : {:<20} + {:<20} * {:<20} = {:<20}",
        "i", "w", "sigma_p", "Ni", "w_p"
    );
    for i in 0..100 {
        let nu_i: f64 = thread_rng().sample(StandardNormal);
        let mut sigma_p = sigma * (tho_p * nu + tho * nu_i).exp();

        if sigma_p < threshold {
            sigma_p = threshold;
        }

        let mut w_p = w + sigma_p * nu_i;

        // if w_p > 1.0 { w_p = w; }

        println!(
            "{:<3} : {:<20} + {:<20} * {:<20} = {:<20}",
            i, w, sigma_p, nu_i, w_p
        );

        sigma = sigma_p;
        w = w_p;
    }
    println!(
        "{:<3} : {:<20} + {:<20} * {:<20} = {:<20}",
        "i", "w", "sigma_p", "Ni", "w_p"
    );
}


fn _test_exploitation() {
    use fluffy_penguin::genetic_algorithm::individual::Specimen;

    println!("Test Exploitation phase.");

    // let mut specimen: Specimen<f32> = Specimen::new(16, 9);
    let specimen_origin: Specimen<f32> = Specimen::new_from_example();
    let mut specimen_mutated: Specimen<f32> = Specimen::new_from_example();

    // let mut specimen_origin: Specimen<f32> = Specimen::new(16, 9);
    // let mut specimen_mutated: Specimen<f32> = specimen_origin.clone();

    for i in 0..10 {
        specimen_mutated.parametric_mutation();
        let mutated_genome = &specimen_mutated.ann.genome;

        println!("Generation {:>3}         ####################################################################", i+1);
        for (origin, mutated) in specimen_origin.ann.genome.iter().zip(mutated_genome) {
            println!(
                " origin: w = {:<15} , sigma = {:<30}",
                origin.w, origin.sigma
            );
            println!(
                "mutated: w = {:<15} , sigma = {:<30}",
                mutated.w, mutated.sigma
            );
            println!("");
        }

        println!("                       ####################################################################\n");
    }
}


fn _test_subnetwork_generation() {
    use fluffy_penguin::genetic_algorithm::individual::Specimen;

    println!("Test subnetwork generation.");

    // let specimen_origin: Specimen<f32> = Specimen::new_from_example();
    // let mut specimen_mutated: Specimen<f32> = Specimen::new_from_example();
    let specimen_origin: Specimen<f32> = Specimen::new(16, 9);
    let mut specimen_mutated: Specimen<f32> = specimen_origin.clone();

    for i in 0..1 {
        println!("Generation {:>3}         ####################################################################", i+1);

        specimen_mutated.structural_mutation(10.0);

        println!("Origin:                    ####################################################################");
        println!("{:#?}", specimen_origin.ann.genome);
        println!("Mutated:                   ####################################################################");
        // println!("{:#?}", specimen_mutated.ann.genome);


        println!("                       ####################################################################\n");
    }
}

fn main() {
    let mut network: Network<f32> = Network::build_from_example();
    println!("Evaluated example output = {:?}", network.evaluate());

    _dev_variation_operator();

    // exploitation();
    // test_exploitation();
    _test_subnetwork_generation();
}
