use cge::{Allele, Network, Node};
use genetic_algorithm::mutation::StructuralMutation;
use rand::distributions::StandardNormal;
use rand::{self, thread_rng, Rng};

pub const LEARNING_RATE_THRESHOLD: f32 = 0.01;


/// A Specimen regroups all the attributes needed by the genetic algorithm of an individual.
#[derive(Clone, Debug, PartialEq)]
pub struct Specimen<T> {
    pub input_size: usize,
    pub output_size: usize,
    /// The ANN.
    pub ann: Network<T>,
    /// Symbolizes how well an individual solves a problem.
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
        self.ann.evaluate().expect("Fail to evaluate this Individual.")
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
    pub fn structural_mutation(&mut self, pm: f32, gin: usize, new_neuron_id: usize) -> Result<(usize,usize), &str> {
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
                Allele::Neuron => {
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

                                let source_id: usize = node.id;
                                let depth: u16 = node.depth;
                                match self.ann.gen_random_jumper_connection(source_id, updated_gin, depth) {
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

                                let sub_network_slice =
                                    &self.ann.genome[node_index..genome_len];

                                let removable_gin_list: Vec<usize> = Network::find_removable_gin_list(&sub_network_slice);

                                if removable_gin_list.len() > 1 {
                                    let removable_gin_index: usize = thread_rng().gen_range(0, removable_gin_list.len());
                                    let removable_gin: usize = removable_gin_list[removable_gin_index];

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
    pub fn crossover(specimen_1: &Specimen<f32>, specimen_2: &Specimen<f32>) -> Specimen<f32> {
        let mut specimen = specimen_1.clone();

        // specimen.ann = Network::crossover(&specimen_1.ann, &specimen_2.ann, specimen_1.fitness, specimen_2.fitness);
        specimen.ann = Network::crossover_2(&specimen_1.ann, &specimen_2.ann, specimen_1.fitness, specimen_2.fitness);

        specimen.update();

        // This is a brand new born offspring so its fitness is null.
        specimen.fitness = 0.0;

        specimen
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
            .expect(&format!("Fail to render Specimen to dot/svg file: {}", file_name));
            // .unwrap_or_else(|e| panic!("failed to execute process: {}", e));
    }
}
