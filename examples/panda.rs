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

    println!("@_dev_population:");

    let population_size: usize = 10;
    let mutation_probability: f32 = 0.1;

    let input_size: usize = 5;
    let output_size: usize = 3;

    let structural_mutation_size: usize = 30;
    let parametric_mutation_size: usize = 100;

    // let input_vector: Vec<f32> = vec![1.0; input_size];
    // let mut pop: Population<f32> = Population::new(
    //     population_size,
    //     input_size,
    //     output_size,
    //     mutation_probability,
    // );

    let input_vector: Vec<f32> = vec![1.0; 2];
    let mut pop: Population<f32> = Population::new_from_example(
        population_size,
        mutation_probability,
    );

    let mut gen: u32 = 0;
    for _ in 0..structural_mutation_size {
        for i in 0..pop.species.len() {
            let mut specimen: &mut Specimen<f32> = &mut pop.species[i];

        // if visualize {
        //     // let file_name: &str = &format!("examples/Gen{:03}_speciment{:02}.dot", gen, i);
        //     let file_name: &str = &format!("examples/Speciment{:02}_Gen{:03}.dot", i, gen);
        //     specimen.render(file_name, "", print_weights);
        // }

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


/// Test all available mutation on the example ANN from the research papers.
fn _test_specimen_mutation(pretty_print: bool, export: bool, print_weights: bool) {
    println!(
        "\n{:^120}",
        "------------------------------------------------------------\n"
    );
    println!("@_test_specimen_mutation:");

    // Example ANN has 2 inputs and 1 output.
    let nbi: usize = 2;
    let mut specimen_origin: Specimen<f32> = Specimen::new_from_example();
    let mut gin: usize = 11;

    // let nbi: usize = 16;
    // let mut specimen_origin: Specimen<f32> = Specimen::new(nbi, 9);
    // let mut gin: usize = specimen_origin.input_size * specimen_origin.output_size;

    let mut specimen_mutated: Specimen<f32> = specimen_origin.clone();

    let input_vector: Vec<f32> = vec![1_f32; nbi];

    {
        if export {
            let file_name: &str = "examples/0rigin.dot";
            let graph_name: &str = "origin";
            // export_visu(&specimen_origin, file_name, graph_name);
            specimen_origin.render(file_name, graph_name, print_weights);
        }

        // println!("");
        specimen_origin.ann.update_input(&input_vector);
        // println!("*Origin: out = {:?}", specimen_origin.ann.evaluate());
        println!("\nOrigin:");
        // specimen_origin.ann.pretty_print();
        if pretty_print {
            Network::pretty_print(&specimen_origin.ann.genome);
        }
        println!("Output = {:?}\n", specimen_origin.evaluate());
    }

    // Mutation probability.
    let pm: f32 = 0.5;
    let generation_size: usize = 20;

    let mut spec_vec: Vec<Specimen<f32>> = Vec::with_capacity(generation_size);

    for i in 0..generation_size {
        gin = specimen_mutated.structural_mutation(pm, gin);

        {
            if export {
                let file_name: &str = &format!("examples/mutated_{:03}.dot", i);
                let graph_name: &str = "mutated";
                // export_visu(&specimen_mutated, file_name, graph_name);
                specimen_mutated.render(file_name, graph_name, print_weights);
            }
        }

        specimen_mutated.ann.update_input(&input_vector);

        println!("\n>> Gen {:>3}: creation", i);
        if pretty_print {
            Network::pretty_print(&specimen_mutated.ann.genome);
        }

        spec_vec.push(specimen_mutated.clone());
        specimen_mutated = specimen_mutated.clone();
        println!("Output = {:?}\n", specimen_mutated.ann.evaluate());

        println!(
            "\n{:^120}",
            "------------------------------------------------------------\n"
        );
    }
}


/// Test crossover operation between specimens.
fn _test_crossover() {
    println!();
    let mut p1: Specimen<f32> = Specimen::new_from_example();
    let mut p2: Specimen<f32> = Specimen::new(2, 1);

    let n1: Network<f32> = Network::_build_parent1_from_example();
    let n2: Network<f32> = Network::_build_parent2_from_example();
    p1.ann = n1;
    p2.ann = n2;

    println!("Parent 1:");
    Network::pretty_print(&p1.ann.genome);
    println!("Output = {:?}\n", p1.evaluate());

    println!("Parent 2:");
    Network::pretty_print(&p2.ann.genome);
    println!("Output = {:?}\n", p2.evaluate());

    let (a1, a2) = Network::align(&p1.ann, &p2.ann);
    Network::pretty_print(&a1.genome);
    // Network::pretty_print(&a2.genome);

}


fn main() {
    let mut network: Network<f32> = Network::build_from_example();
    println!("Evaluated example output = {:?}", network.evaluate());


    // let (pretty_print, visualize, print_weights): (bool, bool, bool) = (true, true, true);
    // _dev_population(pretty_print, visualize, print_weights);

    // _test_exploitation();
    // _test_specimen_mutation(pretty_print, visualize, print_weights);
    _test_crossover();
}
