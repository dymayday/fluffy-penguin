use cge::network::Network;
use cge::node::Node;
use rand::distributions::StandardNormal;
use rand::{thread_rng, Rng};

pub const LEARNING_RATE_THRESHOLD: f32 = 0.01;


/// A Specimen regroups all the attributes needed by the genetic algorithm of an individual.
#[derive(Clone, Debug, PartialEq)]
pub struct Specimen<T> {
    input_size: usize,
    output_size: usize,
    // The ANN.
    pub ann: Network<T>,
    // Symbolizes how well an individual solves a problem.
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
            if sigma_p < LEARNING_RATE_THRESHOLD {
                sigma_p = LEARNING_RATE_THRESHOLD;
            }

            // Compute a new mutated connection weight.
            let w_p: f32 = node.w + sigma_p * nu_i;

            // Assign the new mutated learning rate value to the Node.
            node.sigma = sigma_p;
            // Assign the new mutated weight to the Node.
            node.w = w_p;
        }
    }


    /// Exploration of structures is accomplished by structural mutation which is performed at
    /// larger timescale. It is used to create new species or introduce new structures. From each
    /// of the existing structures, a new structure is formed and added to the existing ones. The
    /// weights of the newly acquired structural parts of the new structure are initialized to zero
    /// so as not to form(get) a new structure whose fitness value is less than its parent.
    ///
    /// pm: is the structural mutation probability and is usually set between 5 and 10%.
    pub fn structural_mutation(&mut self, pm: f32) {
        use cge::node::Allele;

        // Find the unique ID of a potential new Neuron added by the special mutation:
        // 'sub-network addition'.
        let mut new_neuron_id: usize = self.ann.neuron_map.len();

        let mut mutated_genome: Vec<Node<f32>> = Vec::with_capacity(self.ann.genome.len());

        for node in &self.ann.genome {
            match node.allele {
                Allele::Neuron => {
                    if Specimen::to_mutate(pm) {
                        println!("Structural Mutation occuring !");
                        // [TODO]: Add more structural mutation here.
                        {
                            let mut node = node.clone();
                            // N.B.: to add an input connection to the current Neuron
                            // we need to add -1 to the iota value.
                            node.iota -= 1;

                            mutated_genome.push(node);

                            let mut subnetwork: Vec<Node<f32>> =
                                Network::gen_random_subnetwork(new_neuron_id, &self.ann.input_map);
                            println!("New subnetwork: \n{:#?}\n", subnetwork);
                            mutated_genome.append(&mut subnetwork);

                            new_neuron_id += 1;
                        }
                    }
                }
                _ => {
                    mutated_genome.push(node.clone());
                }
            }
        }

        self.ann.genome = mutated_genome;
        self.ann.update_network_attributes();
    }


    /// As part of the structural exploitation, a sub-network has a chance to be added to the
    /// linear genome.
    fn _insert_subnetwork(&mut self, index: usize) {
        // ...
        let _subnetwork: Vec<Node<f32>> =
            Network::gen_random_subnetwork(index, &self.ann.input_map);
    }


    /// Returns if a Neuron Node should be mutated or not by drawing a random number from a uniform
    /// distribution [0, 1) and comparing it with the mutation probability `pm`.
    fn to_mutate(pm: f32) -> bool {
        thread_rng().gen::<f32>() <= pm
    }

    // fn compute_new_neuron_id()
}
