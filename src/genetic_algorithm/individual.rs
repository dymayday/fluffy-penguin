use cge::{Allele, Network, Node};
use genetic_algorithm::mutation::StructuralMutation;
use rand::distributions::StandardNormal;
use rand::{self, thread_rng, Rng};
use std::collections::HashMap;

pub const LEARNING_RATE_THRESHOLD: f32 = 0.01;


/// A Specimen regroups all the attributes needed by the genetic algorithm of an individual.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Specimen<T> {
    pub input_size: usize,
    pub output_size: usize,
    /// The ANN.
    pub ann: Network<T>,
    /// Symbolizes how well an individual solves a problem.
    pub fitness: T,
    pub parents: Vec<Specimen<T>>,
}

impl Specimen<f32> {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Specimen {
            input_size,
            output_size,
            ann: Network::<f32>::new(input_size, output_size),
            fitness: 0.0,
            parents: vec![],
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
            parents: vec![],
        }
    }

    /// Updates a Specimen's attibutes.
    pub fn update(&mut self) {
        self.ann.update();
    }

    /// Update the input vector.
    pub fn update_input(&mut self, input: &[f32]) {
        // assert_eq!(self.input_size.len(), input.len());
        self.ann.update_input(input);
    }

    /// Directly link to the ANN evaluation method.
    pub fn evaluate(&mut self) -> Vec<f32> {
        // pub fn evaluate(&mut self) -> Vec<f32> {
        self.ann
            .evaluate()
            .expect("Fail to evaluate this Individual.")
    }

    /// The exploitation phase researches the optimal weight of each Node in the current artificial
    /// neural network.
    pub fn parametric_mutation(&mut self) {
        // Number of chromosome defining the linear genome.
        let n: f32 = self.ann.genome.len() as f32;

        // The proportionality constant.
        let tau: f32 = 1.0 / (2.0 * n.sqrt()).sqrt();
        let tau_p: f32 = 1.0 / (2.0 * n).sqrt();

        // Denotes a draw from the standard normal distribution.
        let nu: f32 = thread_rng().sample(StandardNormal) as f32;


        for mut node in &mut self.ann.genome {
            // Learning rate value of the current chromosome.
            let sigma: f32 = node.sigma;

            // denotes a separate draw from the standard normal distribution for each node.
            let nu_i: f32 = thread_rng().sample(StandardNormal) as f32;

            // Compute the learning rate matated value.
            let mut sigma_p: f32 = sigma * (tau_p * nu + tau * nu_i).exp() as f32;

            // Since standard deviations very close to zero are unwanted (they will have on average
            // a negligible effect), the following boundary rule is used to force step
            // sizes to be no smaller than a pre-defined threshold.
            // if sigma_p < LEARNING_RATE_THRESHOLD {
            //     sigma_p = LEARNING_RATE_THRESHOLD;
            // }

            // Compute a new mutated connection weight.
            let mut w_p: f32 = node.w + sigma_p * nu_i;

            // // Assign the new mutated learning rate value to the Node.
            // node.sigma = sigma_p;
            // // Assign the new mutated weight to the Node.
            // node.w = w_p;

            // Curtaining the weight and sigma values.
            // if sigma_p < -10.0 || w_p < -10.0 || sigma_p > 10.0 || w_p > 10.0 {
            if w_p < -10.0 || w_p > 10.0 {
                // Do nothing.
            } else {
                // Assign the new mutated learning rate value to the Node.
                node.sigma = sigma_p;
                // Assign the new mutated weight to the Node.
                node.w = w_p;
            }
        }
    }


    /// Exploration of structures is accomplished by structural mutation which is performed at
    /// larger timescale. It is used to create new species or introduce new structures. From each
    /// of the existing structures, a new structure is formed and added to the existing ones. The
    /// weights of the newly acquired structural parts of the new structure are initialized to zero
    /// so as not to form(get) a new structure whose fitness value is less than its parent.
    ///
    /// pm: is the structural mutation probability and is usually set between 5 and 10%.
    ///
    /// This method returns the updated GIN (Global Innovation Number) to keep track of all the new
    /// structures added by a round of structural mutation.
    pub fn structural_mutation(
        &mut self,
        pm: f32,
        gin: usize,
        new_neuron_id: usize,
    ) -> Result<(usize, usize), &str> {
        // Copy the value of the global innovation number to return its updated value by the number
        // of innovation that occured during this mutation cycle.
        let mut updated_gin: usize = gin;

        // Find the unique ID of a potential new Neuron added by the special mutation:
        // 'sub-network addition'.
        // let mut new_neuron_id: usize = self.ann.neuron_map.len();
        let mut new_neuron_id: usize = new_neuron_id;

        let mut mutated_genome: Vec<Node<f32>> = Vec::with_capacity(self.ann.genome.len());

        // let mut mutation_tracker: Vec<String> = Vec::with_capacity(self.ann.genome.len());

        let genome_len: usize = self.ann.genome.len();
        let mut node_index: usize = 0;
        while node_index < genome_len {
            let mut node = self.ann.genome[node_index].clone();

            match node.allele {
                Allele::Neuron { id } => {
                    if Specimen::roll_the_mutation_wheel(pm) {
                        #[allow(unreachable_patterns)]
                        match rand::random::<StructuralMutation>() {
                            StructuralMutation::SubNetworkAddition => {
                                // println!("~~~~~~~~~~~~  StructuralMutation::SubNetworkAddition  ~~~~~~~~~~~~");
                                // Sub-network addition mutation.
                                // mutation_tracker.push("SubNetworkAddition".to_string());

                                let mut iota: i32 = node.iota;

                                // let mut node = node.clone();
                                // N.B.: to add an input connection to the current Neuron
                                // we need to add -1 to the iota value.
                                node.iota -= 1;
                                let depth: u16 = node.depth;

                                // Add the mutated neuron to the mutated genome.
                                mutated_genome.push(node);

                                // Add a new sub-network to the genome.
                                let mut subnetwork: Vec<Node<f32>> = Network::gen_random_subnetwork(
                                    new_neuron_id,
                                    updated_gin + 1,
                                    depth,
                                    &self.ann.input_map,
                                );

                                updated_gin += subnetwork.len();
                                // mutated_genome.append(&mut subnetwork);

                                while node_index + 1 < genome_len {
                                    node_index += 1;
                                    let node = self.ann.genome[node_index].clone();
                                    iota += node.iota;
                                    mutated_genome.push(node);

                                    if iota == 1 {
                                        break;
                                    }
                                }

                                // Add this new sub-network at the end of the the current Neuron
                                // input sub-network.
                                // while node_index + 1 < genome_len {
                                //
                                //     if self.ann.genome[node_index + 1].allele == Allele::Neuron {
                                //         break;
                                //     } else {
                                //         // Add this Neuron's input back to the network.
                                //         node_index += 1;
                                //         let node = self.ann.genome[node_index].clone();
                                //         mutated_genome.push(node);
                                //     }
                                // }

                                mutated_genome.append(&mut subnetwork);

                                new_neuron_id += 1;
                            }

                            StructuralMutation::JumperAddition => {
                                // println!("~~~~~~~~~~~~  StructuralMutation::JumperAddition  ~~~~~~~~~~~~");
                                // Connection addition mutation.
                                // mutation_tracker.push("JumperAddition".to_string());

                                let source_id: usize = id;
                                let depth: u16 = node.depth;
                                match self.ann.gen_random_jumper_connection(
                                    source_id,
                                    updated_gin,
                                    depth,
                                ) {
                                    Some(jumper) => {
                                        // Increase the number of input the current new mutated Neuron has.
                                        node.iota -= 1;
                                        // Add the mutated neuron to the mutated genome.
                                        mutated_genome.push(node);

                                        // Add this new sub-network at the end of the the current Neuron
                                        // input sub-network.
                                        // while node_index + 1 < genome_len {
                                        //     if self.ann.genome[node_index + 1].allele == Allele::Neuron {
                                        //         break;
                                        //     } else {
                                        //         // Add this Neuron's input back to the network.
                                        //         node_index += 1;
                                        //         let node = self.ann.genome[node_index].clone();
                                        //         mutated_genome.push(node);
                                        //     }
                                        // }

                                        // Add the mutated a new jumper connection to the genome
                                        // connecting the current mutated Neuron.
                                        mutated_genome.push(jumper);
                                        updated_gin += 1;
                                    }
                                    None => {
                                        // If there is no possibility to add a new jumper
                                        // connection without breaking everything, we simply push back the current neuron
                                        // unharmed and unmutated.
                                        mutated_genome.push(node);
                                    }
                                }
                            }

                            StructuralMutation::ConnectionRemoval => {
                                // println!("~~~~~~~~~~~~  StructuralMutation::ConnectionRemoval  ~~~~~~~~~~~~");
                                // Connection removal mutation
                                // mutation_tracker.push("ConnectionRemoval".to_string());

                                let sub_network_slice = &self.ann.genome[node_index..genome_len];

                                let removable_gin_list: Vec<usize> =
                                    Network::find_removable_gin_list(&sub_network_slice);

                                if removable_gin_list.len() > 1 {
                                    let removable_gin_index: usize = thread_rng().gen_range(0, removable_gin_list.len());
                                    let removable_gin: usize =
                                        removable_gin_list[removable_gin_index];

                                    // println!("Node updated_gin = {}, removable_gin_list = {:?} x {}", node.gin, removable_gin_list, removable_gin);

                                    node.iota += 1;
                                    mutated_genome.push(node);

                                    for n in sub_network_slice[1..].iter() {
                                        if n.gin == removable_gin {
                                            node_index += 1;
                                            break;
                                        } else {
                                            mutated_genome.push(n.clone());
                                            node_index += 1;
                                        }
                                    }
                                } else {
                                    mutated_genome.push(node);
                                }
                            }

                            _ => {
                                // Unknown structural mutation.
                                println!("Unknown structural mutation behavior draw.");
                            }
                        }
                    } else {
                        // If we don't mutate this Neuron, we simply push it back to the genome,
                        // unharmed ^^'.
                        mutated_genome.push(node);
                    }
                }
                _ => {
                    // Structural mutation only aply on Neuron Node, so we simply push anything that is
                    // not a Neuron back to the genome.
                    mutated_genome.push(node);
                }
            }
            node_index += 1;
        }

        let mut mutated_network: Network<f32> = self.ann.clone();
        mutated_network.genome = mutated_genome;
        mutated_network.update();

        // let mut neuron_list: Vec<usize> = mutated_network.genome.iter().filter(|n| n.allele == Allele::Neuron).map(|n| n.id).collect();
        // neuron_list.sort();
        // println!("Mutated Neuron ids = {:?}", neuron_list);
        // println!("Mutated Genome:");
        // Network::pretty_print(&mutated_network.genome);

        // println!("Mutations : {:#?}", mutation_tracker);

        if mutated_network.is_valid() {
            self.ann = mutated_network;
            Ok((updated_gin, new_neuron_id))
        } else {
            Err("Structural Mutation Failure.")
        }
    }


    /// Returns if a Neuron Node should be mutated or not by drawing a random number from a uniform
    /// distribution [0, 1) and comparing it with the mutation probability `pm`.
    fn roll_the_mutation_wheel(pm: f32) -> bool {
        thread_rng().gen::<f32>() <= pm
    }


    /// Returns the offspring of two Specimens.
    pub fn crossover(father: &Specimen<f32>, mother: &Specimen<f32>, debug: bool) -> Specimen<f32> {
        let mut specimen = father.clone();

        let (father, mother) = Specimen::sort_specimens_genome(&father, &mother, false);
        specimen.ann = Network::crossover_3(
            &father.ann,
            &mother.ann,
            father.fitness,
            mother.fitness,
            debug,
        );

        // specimen.parents = vec![
        //     father.clone(),
        //     mother.clone(),
        // ];

        specimen.update();

        // This is a brand new born offspring so its fitness is null.
        specimen.fitness = 0.0;

        specimen
    }


    /// Sort the second genome according to the order of the first one.
    pub fn sort_specimens_genome(
        specimen_1: &Specimen<f32>,
        specimen_2: &Specimen<f32>,
        debug: bool,
    ) -> (Specimen<f32>, Specimen<f32>) {
        use cge::Allele::Neuron;

        let genome_1 = &specimen_1.ann.genome;
        let genome_2 = &specimen_2.ann.genome;

        let genome_1_len: usize = genome_1.len();
        let genome_2_len: usize = genome_2.len();

        let mut genome_sorted: Vec<Node<f32>> = Vec::with_capacity(genome_1_len + genome_2_len);

        // Let's build a vector containing the GIN of each the Neurons in each genome.
        let n1_gin_vector: Vec<usize> = genome_1
            .iter()
            .filter_map(|n| {
                if let Neuron { .. } = n.allele {
                    Some(n.gin)
                } else {
                    None
                }
            }).collect();

        let n2_gin_vector: Vec<usize> = genome_2
            .iter()
            .filter_map(|n| {
                if let Neuron { .. } = n.allele {
                    Some(n.gin)
                } else {
                    None
                }
            }).collect();


        let ref_specimen;
        let mut other_specimen;

        let ref_genome;
        let other_genome;

        let mut ref_gin_v: Vec<usize>;
        let other_gin_v: Vec<usize>;

        // let ref_neuron_gin_map: HashMap<usize, usize>;
        let other_neuron_gin_map: HashMap<usize, usize>;

        if n1_gin_vector.len() >= n2_gin_vector.len() {
            ref_specimen = specimen_1;
            other_specimen = specimen_2.clone();

            ref_genome = genome_1;
            other_genome = genome_2;

            ref_gin_v = n1_gin_vector;
            other_gin_v = n2_gin_vector;

            // ref_neuron_gin_map = Network::compute_neurons_gin_indices_map(&genome_1);
            other_neuron_gin_map = Network::compute_neurons_gin_indices_map(&genome_2);
        } else {
            ref_specimen = specimen_2;
            other_specimen = specimen_1.clone();

            ref_genome = genome_2;
            other_genome = genome_1;

            ref_gin_v = n2_gin_vector;
            other_gin_v = n1_gin_vector;

            // ref_neuron_gin_map = Network::compute_neurons_gin_indices_map(&genome_2);
            other_neuron_gin_map = Network::compute_neurons_gin_indices_map(&genome_1);
        }

        ref_gin_v.reverse();
        // other_gin_v.reverse();


        let mut slice: Vec<Node<f32>>;


        for ref_neuron_gin in ref_gin_v {
            if debug {
                println!("ref_neuron_gin = {}", ref_neuron_gin);
            }

            let mut gin_already_sorted: Vec<usize> = genome_sorted
                .iter()
                .filter_map(|n| {
                    if let Neuron { .. } = n.allele {
                        Some(n.gin)
                    } else {
                        None
                    }
                }).collect();

            if other_gin_v.contains(&ref_neuron_gin)
                && !gin_already_sorted.contains(&ref_neuron_gin)
            {
                let neuron_idx: usize = *other_neuron_gin_map
                    .get(&ref_neuron_gin)
                    .expect("\n@sort_genome:\n\t>> Fail to lookup ref_neuron_gin.");

                if gin_already_sorted.len() == 0 {
                    slice = other_genome[neuron_idx..].to_vec();
                    genome_sorted.append(&mut slice);
                } else {
                    let end_idx: usize = *other_neuron_gin_map.get(&gin_already_sorted[0])
                        .expect("\n@sort_genome:\n\t>> Fail to lookup ref_neuron_gin in a non empty sorted genome.");

                    slice = other_genome[neuron_idx..end_idx].to_vec();
                    slice.append(&mut genome_sorted);
                    genome_sorted = slice;
                }
            }

            if debug {
                println!("Slice :");
                Network::pretty_print(&genome_sorted);
            }
        }

        if debug {
            println!("\n\n");
            println!("  Ref Genome:");
            Network::pretty_print(&ref_genome);
            println!("Sorted Genome:");
            Network::pretty_print(&genome_sorted);
            println!("Other Genome:");
            Network::pretty_print(&other_genome);
            println!("\n\n");
        }

        other_specimen.ann.genome = genome_sorted;
        assert!(other_specimen.ann.is_valid());

        (ref_specimen.clone(), other_specimen)
    }


    /// Render the Specimen artificial neural network to a dot and svg file.
    pub fn render(&self, file_name: &str, graph_name: &str, print_weights: bool) {
        use std::process::Command;

        let file_name_svg: &str = &String::from(file_name).replace(".dot", ".svg");
        self.ann
            .render_to_dot(file_name, graph_name, print_weights)
            .expect("Fail to render ANN to dot file.");
        Command::new("dot")
            .arg(file_name)
            .arg("-Tsvg")
            .arg("-o")
            .arg(file_name_svg)
            .output()
            .expect(&format!(
                "Fail to render Specimen to dot/svg file: {}",
                file_name
            ));
        // .unwrap_or_else(|e| panic!("failed to execute process: {}", e));
    }


    /// Dump a Specimen into a file using 'Bincode' serialization.
    /// https://github.com/TyOverby/bincode
    pub fn save_to_file(&self, file_name: &str) {
        use bincode::serialize_into;
        use std::fs::File;
        use std::io::BufWriter;
        use utils::create_parent_directory;


        create_parent_directory(file_name).expect(&format!(
            "Fail to create the directory tree of: '{:?}'",
            file_name
        ));
        let stream = BufWriter::new(
            File::create(file_name).expect(&format!("Fail to create file: '{:?}'", file_name)),
        );


        serialize_into(stream, &self).expect("Fail to serialize a Specimen into Bincode file.");
    }


    /// Load a Specimen from a Bincode file.
    /// https://github.com/TyOverby/bincode
    pub fn load_from_file(file_name: &str) -> Self {
        use bincode::deserialize_from;
        use std::fs::File;
        use std::io::BufReader;

        let stream = BufReader::new(
            File::open(file_name).expect(&format!("Fail to open file: '{:?}'", file_name)),
        );

        let specimen: Specimen<f32> =
            deserialize_from(stream).expect("Fail to deserialize a Specimen from Bincode file.");

        specimen
    }
}
