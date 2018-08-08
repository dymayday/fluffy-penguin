//! This example will actually be used during the dev process.

extern crate fluffy_penguin;
extern crate rand;

use fluffy_penguin::cge::Network;
use fluffy_penguin::genetic_algorithm::individual::Specimen;
use rand::{thread_rng, Rng};
// use std::process::Command;


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
fn _dev_population(pretty_print: bool, visualize: bool, print_weights: bool) {
    use fluffy_penguin::genetic_algorithm::population::Population;

    let population_size: usize = 10;
    let input_size: usize = 5;
    let output_size: usize = 3;
    let mutation_probability: f32 = 0.5;

    let structural_mutation_size: usize = 10;
    let parametric_mutation_size: usize = 20;
    let input_vector: Vec<f32> = vec![1.0; input_size];

    let mut pop: Population<f32> = Population::new(
        population_size,
        input_size,
        output_size,
        mutation_probability,
    );

    let mut gen: u8 = 0;
    for _ in 0..structural_mutation_size {
        for i in 0..pop.species.len() {
            let mut specimen: &mut Specimen<f32> = &mut pop.species[i];

            // if visualize {
            //     // let file_name: &str = &format!("examples/Gen{:03}_speciment{:02}.dot", gen, i);
            //     let file_name: &str = &format!("examples/Speciment{:02}_Gen{:03}.dot", i, gen);
            //     specimen.render(file_name, "", print_weights);
            // }
            //
            println!("Gen{:03}: speciment{:02}", gen, i);
            if pretty_print {
                Network::pretty_print(&specimen.ann.genome);
            }
            specimen.ann.update_input(&input_vector);
            println!("Out: {:?}", specimen.evaluate());
            println!("");
        }

        if visualize {
            pop.render("examples/", print_weights);
        }

        for _ in 0..parametric_mutation_size {
            gen += 1;
            pop.exploitation();
        }

        pop.exploration();
        gen += 1;
    }
}


fn _test_exploitation() {
    println!("Test Exploitation phase.");

    let specimen_origin: Specimen<f32> = Specimen::new_from_example();
    let mut specimen_mutated: Specimen<f32> = Specimen::new_from_example();

    // let mut specimen_origin: Specimen<f32> = Specimen::new(16, 9);
    // let mut specimen_mutated: Specimen<f32> = specimen_origin.clone();

    for i in 0..30 {
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
    let nbi: usize = 2;
    let mut specimen_origin: Specimen<f32> = Specimen::new_from_example();

    // let nbi: usize = 16;
    // let mut specimen_origin: Specimen<f32> = Specimen::new(nbi, 9);

    let mut specimen_mutated: Specimen<f32> = specimen_origin.clone();

    let input_vector: Vec<f32> = vec![1_f32; nbi];

    {
        if export {
            let file_name: &str = "examples/0rigin.dot";
            let graph_name: &str = "origin";
            // export_visu(&specimen_origin, file_name, graph_name);
            specimen_origin.render(file_name, graph_name, true);
        }

        // println!("");
        specimen_origin.ann.update_input(&input_vector);
        // println!("*Origin: out = {:?}", specimen_origin.ann.evaluate());
        println!("\nOrigin:");
        // specimen_origin.ann.pretty_print();
        Network::pretty_print(&specimen_origin.ann.genome);
        println!("Output = {:?}\n", specimen_origin.evaluate());
    }

    let generation_size: usize = 20;
    let pm: f32 = 0.1;

    let mut spec_vec: Vec<Specimen<f32>> = Vec::with_capacity(generation_size);

    for i in 0..generation_size {
        specimen_mutated.structural_mutation(pm);

        {
            if export {
                let file_name: &str = &format!("examples/mutated_{}.dot", i);
                let graph_name: &str = "mutated";
                // export_visu(&specimen_mutated, file_name, graph_name);
                specimen_mutated.render(file_name, graph_name, true);
            }
        }

        specimen_mutated.ann.update_input(&input_vector);

        println!("\n>> Gen {:>3}: creation", i);
        Network::pretty_print(&specimen_mutated.ann.genome);

        spec_vec.push(specimen_mutated.clone());
        specimen_mutated = specimen_mutated.clone();
        println!("Output = {:?}\n", specimen_mutated.ann.evaluate());

        println!(
            "\n{:^240}",
            "------------------------------------------------------------\n"
        );
    }

    // println!(
    //     "\n{:^240}",
    //     "------------------------------------------------------------\n"
    // );
    //
    // for i in 0..spec_vec.len() {
    //     let spec = &mut spec_vec[i];
    //
    //     println!("Gen {:>3}:", i);
    //     // spec.ann.pretty_print();
    //     println!("Output = {:?}\n", spec.evaluate());
    // }
}


fn main() {
    let mut network: Network<f32> = Network::build_from_example();
    println!("Evaluated example output = {:?}", network.evaluate());


    let (pretty_print, visualize, print_weights): (bool, bool, bool) = (true, true, true);
    _dev_population(pretty_print, visualize, print_weights);

    // _test_exploitation();
    // _dev_variation_operator();
    // exploitation();
    // test_exploitation();
    // _test_subnetwork_generation(false);
    // _test_specimen_mutation(true);
}
