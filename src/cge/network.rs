//! A flexible encoding method enables one to design an efficient evolutionary
//! method that can evolve both the structures and weights of neural networks.
//! The genome in EANT2 is designed by taking this fact into consideration.
//!
//! A genome in EANT2 is a linear genome consisting of genes (nodes) that can take different forms (alleles).

use activation::TransferFunctionTrait;
use cge::node::{Allele, Node, INPUT_NODE_DEPTH_VALUE, IOTA_INPUT_VALUE};
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::io::Write;

/// The representation of an Artificial Neural Network (ANN) using the Common Genetic Encoding
/// (CGE) (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.8729).
#[derive(Clone, Debug, PartialEq)]
pub struct Network<T> {
    // The linear genome is represented by a vector of Node.
    pub genome: Vec<Node<T>>,
    // Shadowing the genome is the only workaround to fix some immutable with mutable borrow issue.
    // The sizes of the networks should not be an issue though.
    pub shadow_genome: Vec<Node<T>>,
    // This let us map the values of all Input Node. Used during the evaluation phase.
    // Those values are processed by the ReLu transfert function during evaluation time.
    pub input_map: Vec<T>,
    // Neuron value processed by a `Transfer Function`.
    pub neuron_map: Vec<T>,
    // Neuron index lookup table: <genome[i].id, index_location>
    neuron_indices_map: HashMap<usize, usize>,
    // Number of Input in this Network. It has to be a constant value.
    iota_size: usize,
    // The number of Output in this Network. It's a constant value as well.
    omega_size: usize,
}


impl Network<f32> {
    /// Generating the initial linear genome use the grow method by default.
    pub fn new(input_size: usize, outpout_size: usize) -> Self {
        Network::new_simple(input_size, outpout_size)
    }


    /// Starting from simple initial structures is the way it is done by nature and most of the
    /// other evolutionary methods.
    pub fn new_simple(input_size: usize, outpout_size: usize) -> Self {
        // We gather the number of input from the size of the input vector.
        let iota_size: usize = input_size;
        let omega_size: usize = outpout_size;
        // And determine the approximate size of the vectors the network will use.
        let max_vector_size: usize = iota_size * outpout_size;

        let input_map: Vec<f32> = vec![0.0_f32; input_size];

        let mut genome: Vec<Node<f32>> = Vec::with_capacity(max_vector_size);

        {
            // Global Innovation Number is used to keep track of the Nodes to enable crossover
            // later during evolution.
            let mut gin: usize = 1;

            let iota_for_each_neuron: i32 = 1 - iota_size as i32;
            for omega in 0..(omega_size as usize) {
                let neuron: Node<f32> = Node::new(
                    Allele::Neuron,
                    omega,
                    gin,
                    Node::random_weight(),
                    iota_for_each_neuron,
                    0,
                );
                genome.push(neuron);
                gin += 1;

                let mut input_node_vector: Vec<Node<f32>> =
                    Network::gen_input_node_vector(gin, &input_map);
                genome.append(&mut input_node_vector);

                gin += input_size;
            }
        }

        let shadow_genome = genome.clone();
        // let input_map: Vec<f32> = input_vec.clone();
        let neuron_map: Vec<f32> = vec![0.0_f32; omega_size];
        let neuron_indices_map: HashMap<usize, usize> = Network::compute_neuron_indices(&genome);


        Network {
            genome,
            shadow_genome,
            input_map,
            neuron_map,
            neuron_indices_map,
            iota_size,
            omega_size,
        }
    }


    /// Builds and returns a Network from a list of input value using the `grow` method.
    /// For a given maximum depth, the grow method produces linear genomes encoding neural networks
    /// of irregular shape because a node is assigned to a randomly generated neuron node having a
    /// random number of inputs or to a randomly selected input node.
    ///
    /// The grow method, adds more stochastic variation such that the depth of
    /// some (or all) branches may be smaller. Initial structures can also be chosen to be minimal,
    /// which is done in our experiments. This means that an initial network has no hidden layers
    /// or jumper connections, only 1 neuron per output with each of these connected to all inputs.
    /// Starting from simple initial structures is the way it is done by nature and most of the
    /// other evolutionary methods
    ///
    /// There is two important thing to notice here:
    /// * The order of the inputs is particularly important.
    /// * The weights of each input are randomly attributed.
    pub fn new_grow(input_vec: &Vec<f32>, omega_size: usize) -> Self {
        // We gather the number of input from the size of the input vector.
        let iota_size: usize = input_vec.len();
        // And determine the approximate size of the vectors the network will use.
        let max_vector_size: usize = (iota_size * omega_size) as usize;

        let genome: Vec<Node<f32>> = Vec::with_capacity(max_vector_size);

        let shadow_genome = genome.clone();
        let input_map: Vec<f32> = input_vec.clone();
        let neuron_map: Vec<f32> = vec![];
        let neuron_indices_map: HashMap<usize, usize> = Network::compute_neuron_indices(&genome);


        Network {
            genome,
            shadow_genome,
            input_map,
            neuron_map,
            neuron_indices_map,
            iota_size,
            omega_size,
        }
    }


    /// Builds and returns a Network from a list of input value using the `full` method.
    /// This method adds to the linear genome randomly generated neurons connected to all inputs
    /// until a node is at the maximum depth and then adds only random input nodes. This results in
    /// neural networks with symmetric structures where every branch of a tree-based program
    /// equivalent of the linear genome goes to the full maximum depth. In this method, except
    /// neurons at the maximum depth, all neurons are connected to a fixed number of neuron nodes.
    ///
    /// There is two important thing to notice here:
    /// * The order of the inputs is particularly important.
    /// * The weights of each input are randomly attributed.
    pub fn new_full(input_vec: &Vec<f32>, omega_size: usize) -> Self {
        // We gather the number of input from the size of the input vector.
        let iota_size: usize = input_vec.len();
        // And determine the approximate size of the vectors the network will use.
        let max_vector_size: usize = iota_size * omega_size;

        let genome: Vec<Node<f32>> = Vec::with_capacity(max_vector_size);
        let shadow_genome = genome.clone();
        let input_map: Vec<f32> = input_vec.iter().map(|i| i.relu()).collect();
        let neuron_map: Vec<f32> = vec![];
        let neuron_indices_map: HashMap<usize, usize> = Network::compute_neuron_indices(&genome);

        Network {
            genome,
            shadow_genome,
            input_map,
            neuron_map,
            neuron_indices_map,
            iota_size,
            omega_size,
        }
    }


    /// Builds and returns the genome from the research papers we use to implement EANT2.
    pub fn build_from_example() -> Self {
        let input_map = vec![1_f32, 1_f32];
        let neuron_map: Vec<f32> = vec![0.0; 4];
        let genome: Vec<Node<f32>> = vec![
            Node::new(Allele::Neuron, 0, 1, 0.6, -1, 0),
            Node::new(Allele::Neuron, 1, 2, 0.8, -1, 1),
            Node::new(Allele::Neuron, 3, 3, 0.9, -1, 2),
            Node::new(
                Allele::Input,
                0,
                4,
                0.1,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Allele::Input,
                1,
                5,
                0.4,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Allele::Input,
                1,
                6,
                0.5,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(Allele::Neuron, 2, 7, 0.2, -3, 1),
            Node::new(Allele::JumpForward, 3, 8, 0.3, IOTA_INPUT_VALUE, 2),
            Node::new(
                Allele::Input,
                0,
                9,
                0.7,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Allele::Input,
                1,
                10,
                0.8,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(Allele::JumpRecurrent, 0, 11, 0.2, IOTA_INPUT_VALUE, 2),
        ];
        let shadow_genome = genome.clone();
        let neuron_indices_map: HashMap<usize, usize> = Network::compute_neuron_indices(&genome);

        Network {
            genome,
            shadow_genome,
            input_map,
            neuron_map,
            neuron_indices_map,
            iota_size: 2,
            omega_size: 1,
        }
    }
    /// Builds and returns the parent 1's genome from the research papers we use to implement EANT2.
    pub fn _build_parent1_from_example() -> Self {
        let input_map = vec![1_f32, 1_f32];
        let neuron_map: Vec<f32> = vec![0.0; 4];
        let genome: Vec<Node<f32>> = vec![
            Node::new(Allele::Neuron, 0, 1, 0.6, -1, 0),
            Node::new(Allele::Neuron, 1, 2, 0.8, -1, 1),
            Node::new(Allele::Neuron, 3, 7, 0.9, -1, 2),
            Node::new(
                Allele::Input,
                0,
                8,
                0.1,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Allele::Input,
                1,
                9,
                0.4,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Allele::Input,
                1,
                3,
                0.5,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(Allele::Neuron, 2, 4, 0.2, -3, 1),
            Node::new(Allele::JumpForward, 3, 13, 0.3, IOTA_INPUT_VALUE, 2),
            Node::new(
                Allele::Input,
                0,
                5,
                0.7,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Allele::Input,
                1,
                6,
                0.8,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(Allele::JumpRecurrent, 0, 12, 0.2, IOTA_INPUT_VALUE, 2),
        ];
        let shadow_genome = genome.clone();
        let neuron_indices_map: HashMap<usize, usize> = Network::compute_neuron_indices(&genome);

        Network {
            genome,
            shadow_genome,
            input_map,
            neuron_map,
            neuron_indices_map,
            iota_size: 2,
            omega_size: 1,
        }
    }


    /// Builds and returns the parent 2's genome from the research papers we use to implement EANT2.
    pub fn _build_parent2_from_example() -> Self {
        let input_map = vec![1_f32, 1_f32];
        let neuron_map: Vec<f32> = vec![0.0; 3];
        let genome: Vec<Node<f32>> = vec![
            Node::new(Allele::Neuron, 0, 1, 0.8, -1, 0),
            Node::new(Allele::Neuron, 1, 2, 1.0, -2, 1),
            Node::new(Allele::JumpRecurrent, 1, 10, 0.3, IOTA_INPUT_VALUE, 1),
            Node::new(
                Allele::Input,
                0,
                11,
                0.1,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Allele::Input,
                1,
                3,
                0.7,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(Allele::Neuron, 2, 4, 0.9, -1, 1),
            Node::new(
                Allele::Input,
                0,
                5,
                0.5,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Allele::Input,
                1,
                6,
                2.8,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
        ];
        let shadow_genome = genome.clone();
        let neuron_indices_map: HashMap<usize, usize> = Network::compute_neuron_indices(&genome);

        Network {
            genome,
            shadow_genome,
            input_map,
            neuron_map,
            neuron_indices_map,
            iota_size: 2,
            omega_size: 1,
        }
    }




    /// Returns a sub-network composed of one Neuron Node followed by randomly selected
    /// input Node from a vector of input.
    pub fn gen_random_subnetwork(
        neuron_id: usize,
        gin: usize,
        depth: u16,
        input_map: &Vec<f32>,
    ) -> Vec<Node<f32>> {
        let mut subgenome: Vec<Node<f32>> = Vec::with_capacity(1 + input_map.len());
        let gin: usize = gin + 1;

        let mut input_node_vec: Vec<Node<f32>> = Network::gen_input_node_vector(gin, &input_map);
        thread_rng().shuffle(&mut input_node_vec);

        // New hidden neurons are only connected to a subset of inputs, approximately 50%,
        // which makes the search for new structures "more stochastic".
        let input_len_to_pick: usize = input_node_vec.len() / 2 as usize;
        // Or we can randomly pick inputs from the range of input available.
        // let input_len_to_pick: usize = thread_rng().gen_range(1_usize, input_node_vec.len() + 1);

        assert!(input_len_to_pick <= input_node_vec.len());
        input_node_vec = input_node_vec[..input_len_to_pick].to_vec();

        // We compute the number of Input Node this sub-network will have.
        let input_size: usize = input_node_vec.len();
        // And compute the iota value of the Neuron.
        let iota: i32 = 1 - input_size as i32;

        // The initial weight of the first node of a newly added sub-network is set to zero
        // so as not to disturb the performance or behavior of the neural network.
        let neuron: Node<f32> = Node::new(Allele::Neuron, neuron_id, gin, 0.0, iota, depth + 1);
        subgenome.push(neuron);

        // Append all the inputs of our newly created Neuron.
        subgenome.append(&mut input_node_vec);

        // Shrink the sub-network to fit its actual size in memory.
        subgenome.shrink_to_fit();
        subgenome
    }


    /// Returns a vector of Input Node generated from a vector of value.
    fn gen_input_node_vector(neuron_gin: usize, input_vec: &[f32]) -> Vec<Node<f32>> {
        let mut input_node_vector: Vec<Node<f32>> = Vec::with_capacity(input_vec.len());

        let mut gin: usize = neuron_gin;
        for (i, v) in input_vec.iter().enumerate() {
            gin += 1;
            let mut input_node: Node<f32> = Node::new(
                Allele::Input,
                i,
                gin,
                Node::random_weight(),
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            );
            input_node.value = *v;

            input_node_vector.push(input_node);
        }
        input_node_vector
    }


    /// Find the potential Neuron Node with the right depth.
    /// Dsource > Djf and Dtarget =< Djf.
    fn get_potential_neuron_indices_for_jf_node(
        &self,
        source_id: usize,
        depth_source: u16,
    ) -> Vec<usize> {
        let mut indices_v: Vec<usize> = Vec::with_capacity(self.neuron_map.len());

        for i in 0..self.genome.len() {
            let node: &Node<f32> = &self.genome[i];
            match node.allele {
                Allele::Neuron => {
                    if node.depth > depth_source && node.id != source_id {
                        indices_v.push(node.id);
                    }
                }
                _ => {}
            }
        }
        indices_v.shrink_to_fit();
        indices_v
    }


    /// Compute the proper jumper connection Node from a set of rules to not break everything.
    pub fn gen_random_jumper_connection(
        &self,
        source_id: usize,
        gin: usize,
        depth_source: u16,
    ) -> Option<Node<f32>> {
        let mut jumper_kind: Allele;
        let mut potential_target_neuron_indices: Vec<usize>;
        match thread_rng().gen_range(0_usize, 2_usize) {
            0 => {
                jumper_kind = Allele::JumpForward;
                potential_target_neuron_indices =
                    self.get_potential_neuron_indices_for_jf_node(source_id, depth_source);
            }
            _ => {
                jumper_kind = Allele::JumpRecurrent;
                potential_target_neuron_indices =
                    self.neuron_indices_map.keys().map(|x| *x).collect();
            }
        };

        // If there is no possibility to form a new forward connection in the current genome, we
        // fall back to a recurrent connection, which should be in the most cases possible.
        if jumper_kind == Allele::JumpForward && potential_target_neuron_indices.len() < 1 {
            jumper_kind = Allele::JumpRecurrent;
            potential_target_neuron_indices = self.neuron_indices_map.keys().map(|x| *x).collect();
        }

        // Still, we check if there is some possible solution.
        if potential_target_neuron_indices.len() > 0 {
            let jumper_id: usize = *thread_rng()
                .choose(&potential_target_neuron_indices)
                .expect("Fail to draw a jumper connection id to link to an existing Neuron.");
            let jumper: Node<f32> = Node::new(
                jumper_kind,
                jumper_id,
                gin,
                // Set weight to zero in order to not disturb the performance or behavior of the neural network
                0.0,
                IOTA_INPUT_VALUE,
                depth_source + 1,
            );

            Some(jumper)
        } else {
            None
        }
    }


    /// Returns the maximum depth of an artificial neural network.
    pub fn get_max_depth(genome: &[Node<f32>]) -> u16 {
        genome
            .iter()
            .map(|x| x.depth)
            .filter(|x| *x < INPUT_NODE_DEPTH_VALUE)
            .map(|x| x)
            .max()
            .unwrap_or(0_u16)
    }

    /// Compute the indexes of the Neuron in the linear genome so we can find them easily during
    /// the evaluation process. This function is meant to be call only once at init time.
    fn compute_neuron_indices(genome: &Vec<Node<f32>>) -> HashMap<usize, usize> {
        let mut neuron_indices_hashmap: HashMap<usize, usize> =
            HashMap::with_capacity(genome.len() / 2 as usize);
        for i in 0..genome.len() {
            if genome[i].allele == Allele::Neuron {
                neuron_indices_hashmap.insert(genome[i].id, i);
            }
        }
        neuron_indices_hashmap.shrink_to_fit();
        neuron_indices_hashmap
    }


    /// Update the attributes of the network.
    pub fn update_network_attributes(&mut self) {
        self.shadow_genome = self.genome.clone();
        self.neuron_indices_map = Network::compute_neuron_indices(&self.genome);

        self.neuron_map = vec![0.0_f32; self.neuron_indices_map.len()];
    }


    /// Update the input vector.
    pub fn update_input(&mut self, input: &[f32]) {
        assert_eq!(self.input_map.len(), input.len());
        self.input_map = input.to_vec();
    }


    /// Evaluate the linear genome to compute the output of the artificial neural network without decoding it.
    pub fn evaluate(&mut self) -> Vec<f32> {
        // println!("neuron_map: {:?}", self.neuron_map);
        // println!("neuron_indices_map: {:#?}", self.neuron_indices_map);
        let g = self.genome.clone();
        let output: Vec<f32> = self.evaluate_slice(&g);

        // We test here if the evaluation worked smoothly by checking the expected number of
        // output spit out by our artificial neural network.
        assert_eq!(
            output.len(),
            self.omega_size,
            "Evaluated genome output length {} != Expected output length {}",
            output.len(),
            self.omega_size
        );
        output
    }


    /// Evaluate a sub-linear genome to compute the output of an artificial neural sub-network
    /// without decoding it.
    fn evaluate_slice(&mut self, input: &[Node<f32>]) -> Vec<f32> {
        let mut stack: Vec<f32> = Vec::with_capacity(input.len());

        let input_len: usize = input.len();
        // println!("input_len = {}", input_len);

        for i in 0..input_len {
            let mut node: Node<f32> = input[input_len - i - 1].clone();
            // println!("\n{:#?}", node);
            // println!("Stack = {:#?}", stack);

            match node.allele {
                Allele::Input => {
                    stack.push(self.input_map[node.id].relu() * node.w);
                }
                Allele::Neuron => {
                    let neuron_input_len: usize = (1 - node.iota) as usize;
                    let mut neuron_output: f32 = 0.0;
                    for _ in 0..neuron_input_len {
                        // [TODO]: Remove this expect for an unwrap_or maybe ?
                        neuron_output += stack.pop().expect("The evaluated stack is empty.");
                        // neuron_output += stack.pop().unwrap_or(0.0_f32);
                    }

                    node.value = neuron_output;
                    let neuron_index: usize = *self
                        .neuron_indices_map
                        .get(&node.id)
                        .expect(&format!("Fail to lookup the node id = {}", node.id));
                    self.genome[neuron_index].value = neuron_output;

                    // let activated_neuron_value: f32 = node.isrlu(0.1);
                    let activated_neuron_value: f32 = node.relu();
                    // let activated_neuron_value: f32 = node.sigmoids();

                    // Update the neuron value in the neuron_map with its activated value from its
                    // transfert function to be used by jumper connection nodes.
                    self.neuron_map[node.id] = activated_neuron_value;

                    stack.push(activated_neuron_value * node.w);
                }
                Allele::JumpRecurrent => {
                    let recurrent_neuron_value: f32 = self.neuron_map[node.id];
                    stack.push(recurrent_neuron_value * node.w);
                }
                Allele::JumpForward => {
                    // We need to evaluate a slice of our linear genome in a different depth.
                    let forwarded_node_index: usize = *self
                        .neuron_indices_map
                        .get(&node.id)
                        .expect(&format!("Fail to lookup the node id = {}", node.id));

                    let sub_genome_slice: Vec<Node<f32>> =
                        self.shadow_genome[forwarded_node_index..].to_vec();

                    let jf_slice: Vec<Node<f32>> =
                        Network::build_jf_slice(node.id, forwarded_node_index, &sub_genome_slice);

                    // let mut activated_values: Vec<f32> = self.evaluate_slice(&jf_slice).iter().map(|x| x.isrlu(0.1) * node.w).collect();
                    // let mut activated_values: Vec<f32> = self.evaluate_slice(&jf_slice).iter().map(|x| x.relu() * node.w).collect();
                    // stack.append(&mut activated_values);

                    // stack.append(&mut self.evaluate_slice(&jf_slice));
                    // stack.push(self.evaluate_slice(&jf_slice)[0]);
                    // let sum_value: f32 = self.evaluate_slice(&jf_slice).iter().sum::<f32>().isrlu(0.1);
                    stack.push(
                        self.evaluate_slice(&jf_slice)
                            .iter()
                            .sum::<f32>()
                            .isrlu(0.1)
                            * node.w,
                    );
                }
                Allele::NaN => {
                    // Do nothing because the is Not a Node.
                }
            }
        }

        stack
    }


    /// Returns the sub-network corresponding to JumpForward Node to be evaluated as slice of a
    /// Network.
    pub fn build_jf_slice(
        neuron_id: usize,
        neuron_index: usize,
        input_vec: &[Node<f32>],
    ) -> Vec<Node<f32>> {
        let input_len: usize = input_vec.len();
        let mut output_vec: Vec<Node<f32>> = Vec::with_capacity(input_len + 1);

        let mut i: usize = 0;
        let mut iota: i32 = 0;

        while iota != 1 && i < input_len {
            let node: Node<f32> = input_vec[i].clone();

            match node.allele {
                Allele::Neuron => {
                    iota += node.iota;
                }
                _ => {
                    iota += 1;
                }
            }

            output_vec.push(node);

            i += 1;
            if i > input_len {
                println!(
                    "@build_jf_slice:\n\t>> Genome end reached. Looking for N{} at index {}, but we reached index {} and nothing.",
                    neuron_id, neuron_index, i
                );
                Network::pretty_print(input_vec);
                break;
            }
        }

        output_vec.shrink_to_fit();
        output_vec
    }


    /// Returns the input Node vector of a Neuron.
    /// Returns two vectors of Node.
    /// * the first vector are the Node we can actually remove without breaking the linear genome.
    /// * the second one regroups the Nodes we are not allowed to touch.
    pub fn build_input_subnetwork_slice_of_a_neuron(
        neuron_id: usize,
        // neuron_index: usize,
        iota: i32,
        input_vec: &[Node<f32>],
    ) -> (Vec<Node<f32>>, Vec<Node<f32>>) {
        let input_len: usize = input_vec.len();

        let mut disposable_inputs: Vec<Node<f32>> = Vec::with_capacity(input_vec.len());
        let mut untouchable_inputs: Vec<Node<f32>> = Vec::with_capacity(input_vec.len());
        let mut iota: i32 = iota;
        let mut i: usize = 0;


        // As long as we did not walk through all the input from our current Neuron, we keep going.
        while iota != 1 && i < input_len {
            let node: Node<f32> = input_vec[i].clone();

            match node.allele {
                Allele::Neuron => {
                    i += 1;
                    iota += 1;
                    let (mut a, mut b) = Network::build_input_subnetwork_slice_of_a_neuron(
                        node.id,
                        node.iota,
                        &input_vec[i..],
                    );

                    untouchable_inputs.push(node);

                    i += a.len();
                    i += b.len();
                    untouchable_inputs.append(&mut a);
                    untouchable_inputs.append(&mut b);
                }
                _ => {
                    disposable_inputs.push(node);
                    i += 1;
                    iota += 1;
                }
            }

            if i > input_len {
                println!(
                    "@build_input_subnetwork_slice_of_a_neuron:\n\
                     \t>> Sub-genome end reached. Looking for N{} 's inputs.",
                    neuron_id,
                );
                break;
            }
        }

        (disposable_inputs, untouchable_inputs)
    }



    /// Find next Node index with matching GIN.
    fn find_match_gin_node_index(node_index: usize, gin: usize, net1: &Network<f32>, net2: &Network<f32>) {
        // ...
    }


    /// Returns the aligned common parts of two linear genomes.
    pub fn align(network_1: &Network<f32>, network_2: &Network<f32>) -> (Network<f32>, Network<f32>) {

        let max_genome_size: usize = network_1.genome.len() + network_1.genome.len();
        let mut filled_genome_1: Vec<Node<f32>> = Vec::with_capacity(max_genome_size);
        let mut filled_genome_2: Vec<Node<f32>> = Vec::with_capacity(max_genome_size);

        let netw1_len: usize = network_1.genome.len();
        let netw2_len: usize = network_2.genome.len();
        // let mut i: usize = 0;
        // let mut j: usize = 0;
        let (mut i, mut j ): (usize, usize) = (0, 0);
        let mut gin: usize = 1;

        while i < netw1_len && j < netw2_len {
            let n1: &Node<f32> = &network_1.genome[netw1_len - i -1];
            let n2: &Node<f32> = &network_1.genome[netw2_len - j -1];

            if n1.gin == n2.gin {
                gin += 1;
                filled_genome_1.push(n1.clone());
                // filled_genome_2.push(n2.clone());
                i += 1;
                j += 1;
            } else {

                while i < netw1_len && j < netw2_len {
                    let n1: &Node<f32> = &network_1.genome[netw1_len - i -1];
                    // let n2: &Node<f32> = &network_1.genome[netw2_len - j -1];

                    if n1.gin != gin {
                        filled_genome_1.push(n1.clone());
                        i += 1;
                    } else {
                        break;
                    }
                }


                while i < netw1_len && j < netw2_len {
                    // let n1: &Node<f32> = &network_1.genome[netw1_len - i -1];
                    let n2: &Node<f32> = &network_1.genome[netw2_len - j -1];

                    if n2.gin != gin {
                        filled_genome_1.push(n2.clone());
                        j += 1;
                    } else {
                        break;
                    }
                }
            }


        }


        filled_genome_1.reverse();

        let mut netw1 = network_1.clone();
        let mut netw2 = network_2.clone();
        netw1.genome = filled_genome_1;
        netw2.genome = filled_genome_2;
        (netw1, netw2)

        // (Network::build_from_example(), Network::_build_parent2_from_example())
    }


    /// Render an articial neural network to a dot file for a better visualization purpose.
    /// cf.: https://www.graphviz.org/documentation/.
    pub fn render_to_dot(
        &self,
        file_name: &str,
        graph_name: &str,
        print_weight: bool,
    ) -> ::std::io::Result<()> {
        use std::fs::File;
        use std::io::BufWriter;


        let f = File::create(file_name)?;
        {
            let mut writer = BufWriter::new(f);

            // Write header.
            writer.write(format!("digraph {} {{\n", graph_name).as_bytes())?;

            // Pretty printing setup.
            {
                let msg: String = format!(
                    "\trankdir=BT;\n\
                     \tsplines=spline;\n\
                     \tratio=1.0;\n\
                     \tremincross=true;\n\
                     \tnode [fixedsize=false, remincross=true];\
                     \n\n"
                );
                writer.write(msg.as_bytes())?;

                // Print Inputs.
                let msg: String = format!(
                    "\tsubgraph cluster_0 {{\n\
                     \t\tcolor=white;\n\
                     \t\tnode [style=bold, color=orchid, shape=doublecircle];\n\
                     \t"
                );
                writer.write(msg.as_bytes())?;

                for i in 0..self.input_map.len() {
                    let msg: String = format!("I{} ", i);
                    writer.write(msg.as_bytes())?;
                }
                writer.write(";\n\t}\n".as_bytes())?;

                // Print Output Nodes.
                let msg: String = format!(
                    "\tsubgraph cluster_1 {{\n\
                     \t\tcolor=white;\n\
                     \t\tnode [style=bold, color=tomato, shape=doublecircle];\n\
                     \t"
                );
                writer.write(msg.as_bytes())?;

                for i in 0..self.omega_size {
                    let msg: String = format!("N{} ", i);
                    writer.write(msg.as_bytes())?;
                }
                writer.write(";\n\t}\n\n".as_bytes())?;

                let mut empty: bool = true;

                // Paint JF.
                let msg: String = format!(
                    "\tsubgraph cluster_2 {{\n\
                     \t\tcolor=white;\n\
                     \t\tnode [style=solid, color=cornflowerblue, shape=circle];\n\
                     \t"
                );
                writer.write(msg.as_bytes())?;

                for node in &self.genome {
                    if node.allele == Allele::JumpForward {
                        empty = false;
                        let msg: String = format!("JF{} ", node.id);
                        writer.write(msg.as_bytes())?;
                    }
                }
                if !empty {
                    writer.write(";\n".as_bytes())?;
                }
                writer.write("\t}\n\n".as_bytes())?;

                // Paint JR.
                let msg: String = format!(
                    "\tsubgraph cluster_3 {{\n\
                     \t\tcolor=white;\n\
                     \t\tnode [style=solid, color=yellowgreen, shape=circle];\n\
                     \t"
                );
                writer.write(msg.as_bytes())?;

                empty = true;
                for node in &self.genome {
                    if node.allele == Allele::JumpRecurrent {
                        empty = false;
                        let msg: String = format!("JR{} ", node.id);
                        writer.write(msg.as_bytes())?;
                    }
                }
                if !empty {
                    writer.write(";\n".as_bytes())?;
                }
                writer.write("\t}\n\n".as_bytes())?;


                let mut depth_v: Vec<u16> = self.genome.iter().map(|x| x.depth).collect();
                depth_v.sort();
                depth_v.dedup();

                // Paint depth.
                for depth in depth_v {
                    let msg: String = format!("\t{{ rank=same; ");
                    writer.write(msg.as_bytes())?;

                    for node in &self.genome {
                        if node.depth == depth {
                            let msg: String = match node.allele {
                                Allele::JumpRecurrent => format!("JR{} ", node.id),
                                Allele::Neuron => format!("N{} ", node.id),
                                Allele::JumpForward => format!("JF{} ", node.id),
                                _ => format!("I{} ", node.id),
                            };
                            writer.write(msg.as_bytes())?;
                        }
                    }
                    writer.write("};\n".as_bytes())?;
                }
            }

            let input_len: usize = self.genome.len();
            let input: &Vec<Node<f32>> = &self.genome;

            let mut stack: Vec<String> = Vec::with_capacity(input_len);

            for i in 0..input_len {
                let node: &Node<f32> = &input[input_len - i - 1];

                match node.allele {
                    Allele::Input => {
                        stack.push(format!("I{id}", id = node.id));
                        if print_weight {
                            stack.push(format!("[label=\"{w:.3}\"]", w = node.w));
                        } else {
                            stack.push(format!("[label=\"\"]"));
                        }
                    }
                    Allele::JumpForward => {
                        let _msg: String =
                            format!("    {t}{i}[label=\"{t}{i}\"];\n", t = "JF", i = node.id);
                        // writer.write(_msg.as_bytes())?;

                        stack.push(format!("JF{id}", id = node.id));
                        if print_weight {
                            stack.push(format!("[label=\"{w:.3}\"]", w = node.w));
                        } else {
                            stack.push(format!("[label=\"\"]"));
                        }
                    }
                    Allele::JumpRecurrent => {
                        let _msg: String =
                            format!("    {t}{i}[label=\"{t}{i}\"];\n", t = "JR", i = node.id);
                        // writer.write(_msg.as_bytes())?;

                        stack.push(format!("JR{id}", id = node.id));
                        if print_weight {
                            stack.push(format!("[label=\"{w:.3}\"]", w = node.w));
                        } else {
                            stack.push(format!("[label=\"\"]"));
                        }
                    }
                    Allele::Neuron => {
                        let neuron_input_number: usize = (1 - node.iota) as usize;
                        // println!("Dot {} {:#?}", neuron_input_number, node);

                        for _ in 0..neuron_input_number {
                            let msg: String = format!(
                                "    {x} -> N{id}{label};\n",
                                id = node.id,
                                label = stack.pop().expect("No more label in stack."),
                                x = stack.pop().expect("Empty stack."),
                            );

                            writer.write(msg.as_bytes())?;
                        }

                        stack.push(format!("N{id}", id = node.id));
                        if print_weight {
                            stack.push(format!("[label=\"{w:.3}\"]", w = node.w));
                        } else {
                            stack.push(format!("[label=\"\"]"));
                        }
                    }
                    Allele::NaN => {
                        // Do nothing because the is Not a Node.
                    }
                }
            }

            // println!("Stack = {:#?}", stack);
            // Close the graph repsentation.
            writer.write(b"}")?;
        } // the buffer is flushed once writer goes out of scope

        Ok(())
    }


    /// Pretty print the liear genome on a line.
    pub fn pretty_print(genome: &[Node<f32>]) {
        let mut acc: usize = 0;
        for genome_chunk in genome.chunks(20) {
            // Print indices.
            print!("|");
            for i in 0..genome_chunk.len() {
                print!("{:^9}|", format!("[{:^4}]", acc + i));
            }
            println!("");
            acc += genome_chunk.len();

            // The global innovation number.
            print!("|");
            for node in genome_chunk.iter() {
                print!("{:^9}|", format!(" ({:^3})", node.gin));
            }
            println!("");

            // Print Allele and ID.
            print!("|");
            for node in genome_chunk.iter() {
                match node.allele {
                    Allele::Input => print!("{:^9}|", format!(" I{:<3}", node.id)),
                    Allele::Neuron => print!("{:^9}|", format!(" N{:<3}", node.id)),
                    Allele::JumpForward => print!("{:^9}|", format!(" JF{:<3}", node.id)),
                    Allele::JumpRecurrent => print!("{:^9}|", format!(" JR{:<3}", node.id)),
                    Allele::NaN => print!("{:^9}|", format!(" X{:<3}", 'x')),
                }
            }
            println!("");

            // Print depths.
            print!("|");
            for node in genome_chunk.iter() {
                print!("{:^9}|", format!("d{:<2}", node.depth));
            }
            println!("");

            // Print weights.
            print!("|");
            for node in genome_chunk.iter() {
                print!("{:^9}|", format!("w{:.3}", node.w));
            }
            println!("");

            // Print iotas.
            print!("|");
            for node in genome_chunk.iter() {
                print!("{:^9}|", format!("{:^3}", node.iota));
            }
            println!("");
        }
    }
}
