//! This example will actually be used during the dev process.

extern crate fluffy_penguin;
extern crate rand;

use fluffy_penguin::cge::Network;
use fluffy_penguin::genetic_algorithm::individual::Specimen;
use fluffy_penguin::genetic_algorithm::Population;
use rand::{thread_rng, Rng};
use std::fs;
use std::path::Path;
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
    use fluffy_penguin::genetic_algorithm::Population;

    println!("@_dev_population:");

    let population_size: usize = 10;
    let mutation_probability: f32 = 0.1;

    // let input_size: usize = 5;
    // let output_size: usize = 3;

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
    let mut pop: Population<f32> =
        Population::new_from_example(population_size, mutation_probability);

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
    let mut nn_id: usize = 4;

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
        let (gin_tmp, nn_id_tmp) = specimen_mutated
            .structural_mutation(pm, gin, nn_id)
            .unwrap();
        gin = gin_tmp;
        nn_id = nn_id_tmp;

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

    let nbi: usize = 2;
    let input_vector: Vec<f32> = vec![1_f32; nbi];

    let mut sp1: Specimen<f32> = Specimen::new_from_example();
    let mut sp2: Specimen<f32> = Specimen::new(2, 1);

    let n1: Network<f32> = Network::_build_parent1_from_example();
    let n2: Network<f32> = Network::_build_parent2_from_example();
    sp1.ann = n1;
    sp2.ann = n2;

    println!("Parent 1:");
    Network::pretty_print(&sp1.ann.genome);
    println!("Output = {:?}\n", sp1.evaluate());

    println!("Parent 2:");
    Network::pretty_print(&sp2.ann.genome);
    println!("Output = {:?}\n", sp2.evaluate());

    // let crossover_specimen = Specimen::crossover(&sp1, &sp2);

    // println!("\n>> Crossover: ");
    // Network::pretty_print(&crossover_specimen.ann.genome);

    // let crossover_specimen = Specimen::crossover(&sp2, &sp1);
    // Network::pretty_print(&crossover_specimen.ann.genome);

    // Network::align(&sp1.ann, &sp2.ann);

    let mut offspring: Specimen<f32> = Specimen::crossover(&sp1, &sp2, false);
    println!("\n>> Offspring:");
    Network::pretty_print(&offspring.ann.genome);

    offspring.ann.update_input(&input_vector);
    println!("Output = {:?}\n", offspring.ann.evaluate());
}


fn _test_population_crossover(pretty_print: bool, export: bool, print_weights: bool) {
    use fluffy_penguin::cge::Allele;
    println!();

    let population_size: usize = 2;
    let input_size: usize = 2;
    let output_size: usize = 1;
    let mutation_probability: f32 = 0.5;

    loop {
        let mut population: Population<f32> = Population::new(
            population_size,
            input_size,
            output_size,
            mutation_probability,
        );

        println!("{:#^240}", "");
        println!("{:#^240}", "    Init population:    ");
        println!("{:#^240}", "");
        for (i, specimen) in population.species.iter().enumerate() {
            if pretty_print {
                println!("Specimen {}", i);
                Network::pretty_print(&specimen.ann.genome);
            }
            if export {
                let file_name: &str = &format!("examples/specimen-{:03}_aaa.dot", i);
                let graph_name: &str = "initial";
                specimen.render(file_name, graph_name, print_weights);
            }
        }
        println!();

        let structural_mutation_size: usize = 20;

        for smi in 0..structural_mutation_size {
            population.exploration();

            println!("~~~~~~~~~~~~ After Structural Mutation:");
            for (i, specimen) in population.species.iter().enumerate() {
                println!("Specimen {}", i);
                Network::pretty_print(&specimen.ann.genome);
                println!(
                    "Output = {:?}",
                    Network::pseudo_evaluate_slice(&specimen.ann.genome)
                );
            }
            println!(":After Structural Mutation ~~~~~~~~~~~~");
            println!("\n");

            population.evolve();

            println!("\n\n\t///  Evolution {}  \\\\\\", smi + 1);
            for (i, offspring) in population.species.iter().enumerate() {
                println!("Offspring {}", i);

                let mut neuron_list: Vec<usize> = offspring
                    .ann
                    .genome
                    .iter()
                    .filter_map(|n| {
                        if let Allele::Neuron { id } = n.allele {
                            Some(id)
                        } else {
                            None
                        }
                    }).collect();
                neuron_list.sort();
                println!("Neuron ids = {:?}", neuron_list);

                if pretty_print {
                    Network::pretty_print(&offspring.ann.genome);
                }
                if export {
                    let file_name: &str =
                        &format!("examples/specimen-{:03}_evolved-{:03}.dot", i, smi);
                    let graph_name: &str = "mutated";
                    // export_visu(&specimen_mutated, file_name, graph_name);
                    offspring.render(file_name, graph_name, print_weights);
                }
                println!(
                    "Output = {:?}",
                    Network::pseudo_evaluate_slice(&offspring.ann.genome).unwrap()
                );
            }
            println!("\t\\\\\\  Evolution {}  ///\n\n", smi + 1);
            println!();
        }

        println!("\n\n\n");
    }
}


fn _test_population_selection(pretty_print: bool, export: bool, print_weights: bool) {
    let population_size: usize = 32;
    let input_size: usize = 8;
    let output_size: usize = 9;
    let mutation_probability: f32 = 0.1;

    let mutation_size: usize = 100;

    // let mut population: Population<f32> =
    // Population::new(population_size, input_size, output_size, mutation_probability);
    // Population::new_from_example(population_size, mutation_probability);


    let mut loop_counter: usize = 0;
    loop {
        let mut file_to_remove: Vec<String> = Vec::with_capacity(population_size);
        let mut population: Population<f32> = Population::new(
            population_size,
            input_size,
            output_size,
            mutation_probability,
        );

        for _smi in 0..mutation_size {
            loop_counter += 1;
            println!("Loop counter = {:>6}", loop_counter);

            // println!("Init population:");
            for i in 0..population.species.len() {
                let mut specimen: &mut Specimen<f32> = &mut population.species[i];
                specimen.fitness = thread_rng().gen_range(0_i32, 101_i32) as f32;

                if pretty_print {
                    println!("Specimen {}", i);
                    Network::pretty_print(&specimen.ann.genome);
                }

                if export {
                    let file_name: String = format!("tmp/specimen-{:04}.dot", specimen.fitness);

                    let graph_name: &str = "initial";
                    specimen.render(&file_name, graph_name, print_weights);

                    file_to_remove.push(file_name);
                }
            }
            // println!();
            //
            // let lowest_fitness: f32 = *population.species
            //     .iter()
            //     .map(|s| s.fitness)
            //     .collect::<Vec<f32>>()
            //     .iter()
            //     .min_by( |x, y| x.partial_cmp(y).unwrap() )
            //     .unwrap_or(&0.0);
            //
            // population.sort_species_by_fitness();
            // let sus_selected = &population.species;
            // for i in 0..sus_selected.len() {
            //     // println!(" {:>4} : {:<4}", *&mut population.species[i].fitness as i32, sus_selected[i].fitness as i32);
            //     println!(" {:>4} : {:<4}", "", sus_selected[i].fitness + lowest_fitness.abs());
            // }

            if _smi % 10 == 0 {
                // || true {
                population.exploration();
            } else {
                population.exploitation();
            }
            population.evolve();
        }

        if export {
            file_to_remove.sort();
            file_to_remove.dedup();

            for file_name in file_to_remove {
                let file_path = Path::new(&file_name);
                fs::remove_file(&file_path).expect(&format!("Fail to remove {:?}", file_path));

                let svg_file_name: String = file_name.replace(".dot", ".svg");
                let file_path = Path::new(&svg_file_name);
                fs::remove_file(&file_path).expect(&format!("Fail to remove {}", svg_file_name));
            }
        }
    }
}


/// To concentrate on a bug fix we need to focus on specimens that do fuckup instead of waiting for
/// a random bug to appear.
fn _test_on_defective_specimens() {
    let father: Specimen<f32> = Specimen::load_from_file("tmp/father.bc");
    let mother: Specimen<f32> = Specimen::load_from_file("tmp/mother.bc");

    // let father: Specimen<f32> = Specimen::load_from_file("tmp/father-father.bc");
    // let mother: Specimen<f32> = Specimen::load_from_file("tmp/mother-mother.bc");

    println!("Failed Father's Grand-Father:");
    Network::pretty_print(&father.parents[0].ann.genome);
    println!("Failed Father's Grand-Mother:");
    Network::pretty_print(&father.parents[1].ann.genome);
    println!("Failed Parent:");
    Network::pretty_print(&father.ann.genome);

    let father_bis = Specimen::crossover(&father.parents[0], &father.parents[1], true);
    println!("Father BIS:");
    Network::pretty_print(&father_bis.ann.genome);
    father_bis.render("tmp/father_bis.dot", "father", false);


    // father.render("tmp/father.dot", "father", false);
    // mother.render("tmp/mother.dot", "mother", false);
    // father.parents[0].render("tmp/father-father.dot", "father_father", false);
    // father.parents[1].render("tmp/father-mother.dot", "father_mother", false);
    // mother.parents[0].render("tmp/mother-father.dot", "mother_father", false);
    // mother.parents[1].render("tmp/mother-mother.dot", "mother_mother", false);


    let mut offspring = Specimen::crossover(&father, &mother, true);
    println!("Failed Offspring:");
    Network::pretty_print(&offspring.ann.genome);
    println!("Offspring is valid ? {}", offspring.ann.is_valid());
}


fn _test_crossover3_on_defective_specimens() {
    let father: Specimen<f32> = Specimen::load_from_file("tmp/father.bc");
    let mother: Specimen<f32> = Specimen::load_from_file("tmp/mother.bc");

    println!("Failed Father's Grand-Father:");
    Network::pretty_print(&father.parents[0].ann.genome);
    println!("Failed Father's Grand-Mother:");
    Network::pretty_print(&father.parents[1].ann.genome);
    println!("Failed Parent:");
    Network::pretty_print(&father.ann.genome);

    println!(
        "\n{:^120}",
        "µµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµ\n"
    );

    // let father_bis = Specimen::crossover(&father.parents[0], &father.parents[1], true);
    let father_bis_network = Network::crossover_3(
        &father.parents[0].ann,
        &father.parents[1].ann,
        father.parents[0].fitness,
        father.parents[1].fitness,
        true,
    );
    println!("Father BIS:");
    Network::pretty_print(&father_bis_network.genome);
    let mut father_bis = father.clone();
    father_bis.ann = father_bis_network;
    father_bis.render("tmp/father_bis.dot", "father", false);


    // let mut offspring = Specimen::crossover(&father, &mother, true);
    // println!("Failed Offspring:");
    // Network::pretty_print(&offspring.ann.genome);
    // println!("Offspring is valid ? {}", offspring.ann.is_valid());
}


fn main() {
    let (pretty_print, visualize, print_weights): (bool, bool, bool) = (false, false, false);
    // _dev_population(pretty_print, visualize, print_weights);

    // _test_exploitation();
    // _test_specimen_mutation(pretty_print, visualize, print_weights);
    // _test_crossover();
    // _test_population_crossover(pretty_print, visualize, print_weights);
    _test_population_selection(pretty_print, visualize, print_weights);
    // _test_on_defective_specimens();
    // _test_crossover3_on_defective_specimens();
}
