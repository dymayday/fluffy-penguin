//! This example will actually be used during the dev process.

extern crate fluffy_penguin;
extern crate rand;

use fluffy_penguin::cge::network::Network;
use rand::{thread_rng, Rng};
use std::process::Command;
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


fn _test_specimen_mutation(export: bool) {
    use fluffy_penguin::genetic_algorithm::individual::Specimen;

    let nbi: usize = 2;
    let mut specimen_origin: Specimen<f32> = Specimen::new_from_example();
    // let nbi: usize = 16;
    // let mut specimen_origin: Specimen<f32> = Specimen::new(nbi, 9);
    let mut specimen_mutated: Specimen<f32> = specimen_origin.clone();

    let input_vector: Vec<f32> = vec![1_f32; nbi];

    {
        let file_name: &str = "examples/0rigin.dot";
        let file_name_svg: &str = "examples/0rigin.svg";
        let graph_name: &str = "origin";


        if export {
            specimen_origin
                .ann
                .render_to_dot(file_name, graph_name)
                .expect("Fail to render ANN to dot file.");
            Command::new("dot")
                .arg(file_name)
                .arg("-Tsvg")
                .arg("-o")
                .arg(file_name_svg)
                .output()
                .unwrap_or_else(|e| panic!("failed to execute process: {}", e));
        }

        // println!("");
        specimen_origin.ann.update_input(&input_vector);
        // println!("*Origin: out = {:?}", specimen_origin.ann.evaluate());
        println!("\nOrigin:");
        specimen_origin.ann.pretty_print();
        println!("Output = {:?}\n", specimen_origin.evaluate());
    }

    let generation_size: usize = 10;
    let pm: f32 = 0.5;

    let mut spec_vec: Vec<Specimen<f32>> = Vec::with_capacity(generation_size);
    specimen_mutated.structural_mutation(pm);

    for i in 0..generation_size {
        specimen_mutated.structural_mutation(pm);


        {
            let file_name: &str = &format!("examples/mutated_{}.dot", i);
            let file_name_svg: &str = &format!("examples/mutated_{}.svg", i);
            let graph_name: &str = "mutated";


            if export {
                specimen_mutated
                    .ann
                    .render_to_dot(file_name, graph_name)
                    .expect("Fail to render ANN to dot file.");

                Command::new("dot")
                    .arg(file_name)
                    .arg("-Tsvg")
                    .arg("-o")
                    .arg(file_name_svg)
                    .output()
                    .unwrap_or_else(|e| panic!("failed to execute process: {}", e));
            }
        }

        specimen_mutated.ann.update_input(&input_vector);
        // println!("Gen {:>3}: out = {:?}", i+1, specimen_mutated.ann.evaluate());
        println!("Gen {:>3}: creation", i);
        // println!("{:#?}", specimen_mutated.ann.genome);

        spec_vec.push(specimen_mutated.clone());
        specimen_mutated = specimen_mutated.clone();
    }


    for i in 0..spec_vec.len() {
        let spec = &mut spec_vec[i];
        println!("\nGen {:>3}:", i);
        spec.ann.pretty_print();
        println!("Output = {:?}\n", spec.evaluate());
    }
}


fn main() {
    let mut network: Network<f32> = Network::build_from_example();
    println!("Evaluated example output = {:?}", network.evaluate());

    _dev_variation_operator();

    // exploitation();
    // test_exploitation();
    // _test_subnetwork_generation(false);
    _test_specimen_mutation(true);
}
