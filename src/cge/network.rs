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
    // pub neuron_value_map: HashMap<usize, T>,
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
                let mut input_node_vector: Vec<Node<f32>> =
                    Network::gen_input_node_vector(gin, &input_map);

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

                genome.append(&mut input_node_vector);

                gin += input_size;
            }
        }

        let shadow_genome = genome.clone();
        // let input_map: Vec<f32> = input_vec.clone();
        let neuron_map: Vec<f32> = vec![0.0_f32; omega_size];
        // let neuron_value_map: HashMap<usize,f32> = HashMap::with_capacity(omega_size);
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
        new_neuron_gin: usize,
        depth: u16,
        input_map: &Vec<f32>,
    ) -> Vec<Node<f32>> {

        let mut subgenome: Vec<Node<f32>> = Vec::with_capacity(1 + input_map.len());
        let gin: usize = new_neuron_gin;

        let mut input_node_vec: Vec<Node<f32>> = Network::gen_input_node_vector(gin + 1, &input_map);
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

        // Update each Input Node's GIN with their proper values.
        for i in 0..input_node_vec.len() {
            input_node_vec[i].gin = gin + 1 + i;
        }

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
                gin + 1,
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


    /// Update a Network.
    pub fn update(&mut self) {
        self.update_network_attributes();
    }

    /// Update the attributes of the network.
    pub fn update_network_attributes(&mut self) {
        self.shadow_genome = self.genome.clone();
        self.neuron_indices_map = Network::compute_neuron_indices(&self.genome);

        // self.neuron_map = vec![0.0_f32; self.neuron_indices_map.len()];
        
        let neuron_id_max: usize = 
            *self.genome
            .iter()
            .filter(|n| n.allele == Allele::Neuron)
            .map(|n| n.id)
            .collect::<Vec<usize>>()
            .iter()
            .max()
            .unwrap();
        self.neuron_map = vec![0.0_f32; neuron_id_max + 1];
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
        // assert_eq!(
        //     output.len(),
        //     self.omega_size,
        //     "Evaluated genome output length {} != Expected output length {}",
        //     output.len(),
        //     self.omega_size
        // );
        output
    }


    /// Evaluate a sub-linear genome to compute the output of an artificial neural sub-network
    /// without decoding it.
    fn evaluate_slice(&mut self, input: &[Node<f32>]) -> Vec<f32> {
    // fn evaluate_slice(&mut self, input: &[Node<f32>]) -> Result<Vec<f32>, &str> {
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
                        // let neuron_output += match stack.pop() {
                        //     Ok(v) => v,
                        //     Err(_) => return Err("The evaluated stack is empty.")
                        // };
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
        // Ok(stack)
    }


    /// Pseudo evaluation.
    pub fn pseudo_evaluate_slice(input: &[Node<f32>]) -> Option<Vec<f32>> {
        let mut stack: Vec<f32> = Vec::with_capacity(input.len());

        let input_len: usize = input.len();


        for i in 0..input_len {
            let mut node: &Node<f32> = &input[input_len - i - 1];

            match node.allele {
                Allele::Input | Allele::JumpForward | Allele::JumpRecurrent => {
                    stack.push(0.0);
                }
                Allele::Neuron => {
                    let neuron_input_len: usize = (1 - node.iota) as usize;
                    for _ in 0..neuron_input_len {
                        // [TODO]: Remove this expect for an unwrap_or maybe ?
                        // stack.pop().expect("The pseudo evaluated stack is empty.");
                        // neuron_output += stack.pop().unwrap_or(0.0_f32);
                        match stack.pop() {
                            Some(_) => {},
                            None => return None,
                        }
                    }

                    stack.push(0.0);
                }
                _ => {}


            }
        }

        Some(stack)
    }



    /// Returns if a Network is considered valid.
    pub fn is_valid(&mut self) -> bool {
        let inputs: Vec<f32> = vec![1.0; self.input_map.len()];

        self.update_input(&inputs);
        let output: Vec<f32> = self.evaluate();

        if output.len() != self.omega_size {
            println!("output.len() {} != {} self.omega_size", output.len(), self.omega_size);
            return false;
        }

        let iota_sum: i32 = self.genome.iter().map(|n| n.iota).collect::<Vec<i32>>().iter().sum();
        if self.omega_size as i32 != iota_sum {
            println!("iota_sum {} != {} self.omega_size", iota_sum, self.omega_size);
            return false;
        }
        true
    }


    /// Find and returns the proper iota value after a crossover.
    /// The updated number of input of a Node affected by a crossover operator follows the
    /// equation:
    /// n(s1 × s2 ) = n(s1 ) + n(s2) − n(s1 ∩ s2 )
    fn _compute_iota_after_crossover(input_1: &[Node<f32>], input_2: &[Node<f32>]) -> i32 {
        let nbi_1: i32 = 1 - input_1[0].iota;
        let nbi_2: i32 = 1 - input_2[0].iota;

        // Now we need to find the number of input Node common to both subnetwork we have as
        // inputs.
        // let gin_1: Vec<usize> = input_1[1..].iter().map(|n| n.gin).collect();
        let gin_2: Vec<usize> = input_2[1..].iter().map(|n| n.gin).collect();

        let mut common_node: Vec<Node<f32>> = Vec::with_capacity(input_1.len());

        for i in 0..input_1.len() {
            let gin_1_tmp: usize = input_1[i].gin;
            if gin_2.contains(&gin_1_tmp) {
                common_node.push(input_1[i].clone());
            }
        }

        common_node.reverse();

        let common_node_output: Vec<f32> = Network::pseudo_evaluate_slice(&common_node).unwrap_or(vec![]);


        1 - (nbi_1 + nbi_2 - common_node_output.len() as i32)
    }


    /// Build and returns the sub-network of a genome's slice.
    fn _build_subnetwork_slice(input: &[Node<f32>]) -> Vec<Node<f32>> {
        let input_len: usize = input.len();
        let mut subnetwork: Vec<Node<f32>> = Vec::with_capacity(input_len);


        let mut iota: i32 = input[0].iota;
        let mut i: usize = 1;

        subnetwork.push(input[0].clone());

        while iota < 1 && i < input_len {
            iota += input[i].iota;
            subnetwork.push(input[i].clone());
            i += 1;
        }

        subnetwork.shrink_to_fit();
        subnetwork
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


    /// Find removable GIN from a Neuron.
    pub fn find_removable_gin_list(genome: &[Node<f32>]) -> Vec<usize> {

        let genome_len: usize = genome.len();
        let mut removable_gin: Vec<usize> = Vec::with_capacity(genome_len);

        let mut iota: i32 = genome[0].iota;
        let mut i: usize = 1;


        while i < genome_len && iota != 1 {
            
            let node: Node<f32> = genome[i].clone();

            match node.allele {
                Allele::Neuron => {
                    // let mut gin_list: Vec<usize> = Network::find_removable_gin_list(&genome[i..]);
                    // removable_gin.append(&mut gin_list);
                    break;
                },
                Allele::Input | Allele::JumpForward | Allele::JumpRecurrent => {
                    removable_gin.push(node.gin);
                    iota += 1;
                },
                _ => { break; }
            }
            i += 1;
        }

        removable_gin
    }


    /// Returns the input Node vector of a Neuron.
    /// Returns two vectors of Node.
    /// * the first vector cpntains the Nodes we can actually remove without breaking the linear genome.
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



    /// Crossover.
    pub fn crossover(network_1: &Network<f32>, network_2: &Network<f32>, fitness_1: f32, fitness_2: f32) -> Network<f32> {

        let default_network: &Network<f32>;
        if fitness_2 > fitness_1 {
            // default_network = network_2.clone();
            default_network = &network_2;
        } else {
            // default_network = network_1.clone();
            default_network = &network_1;
        }
        let (netw_1, netw_2) = Network::align(&network_1, &network_2).unwrap_or((default_network.clone(), default_network.clone()));

        let mut netw_crossovered = netw_1.clone();

        let netw_len: usize = netw_1.genome.len();
        for i in 0..netw_len {
            let n1: &Node<f32> = &netw_1.genome[i];
            let n2: &Node<f32> = &netw_2.genome[i];

            if n1.allele == n2.allele {
                let pick_from: [f32; 2] = [n1.w, n2.w];
                let rnd_weight_ref: &f32 = thread_rng().choose(&pick_from).unwrap_or(&n1.w);
                netw_crossovered.genome[i].w = *rnd_weight_ref;

            } else {
                if n1.allele != Allele::NaN {
                    netw_crossovered.genome[i] = n1.clone();
                } else if n2.allele != Allele::NaN {
                    netw_crossovered.genome[i] = n2.clone();
                } else {
                    assert_eq!(n1.allele, Allele::NaN);
                    assert_eq!(n2.allele, Allele::NaN);
                }
            }
        }
        netw_crossovered.update();
        netw_crossovered
    }

    /// Aligning.
    pub fn align(network_1: &Network<f32>, network_2: &Network<f32>) -> Result<(Network<f32>, Network<f32>), ()> {
    // pub fn align(network_1: &Network<f32>, network_2: &Network<f32>) -> (Network<f32>, Network<f32>) {

        println!("\n\n\n================================================   Aligning...   ================================================\n");

        let arn_1: Vec<Node<f32>> = Network::_compute_aligned_arn(&network_1.genome, &network_2.genome);
        let mut arn_2: Vec<Node<f32>> = Network::_compute_aligned_arn(&network_2.genome, &network_1.genome);


        println!("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   ARN 1:");
        Network::pretty_print(&arn_1);
        //
        // println!("ARN 2:");
        // Network::pretty_print(&arn_2);

        arn_2 = Network::sort_arn(&arn_1, &arn_2);

        println!("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   ARN 2 sorted:");
        Network::pretty_print(&arn_2);

        println!();

        let arn_1_updated = Network::arn_update_iota(&arn_1, &arn_2);
        println!();
        let arn_2_updated = Network::arn_update_iota(&arn_2, &arn_1);


        // Check iota values between the 2 ARN.
        let iota_1: Vec<i32> = arn_1_updated.iter().map(|n| n.iota).collect();
        let iota_2: Vec<i32> = arn_2_updated.iter().map(|n| n.iota).collect();

        println!("ARN 1 with updated iota values:");
        Network::pretty_print(&arn_1_updated);

        let iota_sum_1: i32 = iota_1.iter().sum();
        // let out_1 = Network::pseudo_evaluate_slice(&arn_1_updated);
        // println!("Iota = {}, Output = {:?}", iota_sum_1, out_1);

        // assert_eq!(network_1.omega_size as i32, iota_sum_1, "iota and expected output length mismatch.");
        if network_1.omega_size as i32 != iota_sum_1 {
            // println!("\n\n");
            println!("iota 1 and expected output length mismatch. {} != {}", network_1.omega_size as i32, iota_sum_1);
            // Network::pretty_print(&network_1.genome);
            // Network::pretty_print(&arn_1_updated);
            // Network::pretty_print(&arn_2_updated);
            // Network::pretty_print(&network_2.genome);
            // println!("\n\n");
            // println!("Out = {:?}", arn_1.clone().evaluate());
            // return Err(())
        }

        println!("ARN 2 with updated iota values:");
        Network::pretty_print(&arn_2_updated);

        let iota_sum_2: i32 = iota_2.iter().sum();
        // let out_2 = Network::pseudo_evaluate_slice(&arn_2_updated);
        // println!("Iota = {}, Output = {:?}", iota_sum_2, out_2);

        // assert_eq!(network_2.omega_size as i32, iota_sum_2, "iota and expected output length mismatch.");
        if network_2.omega_size as i32 != iota_sum_2 {
            // println!("\n\n");
            println!("iota 2 and expected output length mismatch. {} != {}", network_2.omega_size as i32, iota_sum_2);
            // Network::pretty_print(&network_1.genome);
            // Network::pretty_print(&arn_1_updated);
            // Network::pretty_print(&arn_2_updated);
            // Network::pretty_print(&network_2.genome);
            // println!("\n\n");
            // return Err(())
        }

        // println!("\n");

        // assert_eq!(iota_1, iota_2, "Iota values between ARN diverge.");
        if iota_1 != iota_2 {
            println!("iota1 != iota2. {:?} != {:?}", iota_1, iota_2);
            // return Err(())
        }

        let mut netw_1_aligned = network_1.clone();
        let mut netw_2_aligned = network_2.clone();

        netw_1_aligned.genome = arn_1_updated;
        netw_2_aligned.genome = arn_2_updated;

        Ok((netw_1_aligned, netw_2_aligned))
    }


    fn _compute_aligned_arn(genome_1: &[Node<f32>], genome_2: &[Node<f32>]) -> Vec<Node<f32>> {

        let max_genome_size: usize = genome_1.len() + genome_2.len();
        let netw1_len: usize = genome_1.len();
        let netw2_len: usize = genome_2.len();

        let n1_gin_vector: Vec<usize> = genome_1.iter().map(|n| n.gin).collect();
        let n2_gin_vector: Vec<usize> = genome_2.iter().map(|n| n.gin).collect();

        let mut arn: Vec<Node<f32>> = Vec::with_capacity(max_genome_size);

        let (mut i, mut j ): (usize, usize) = (0, 0);
        while i < netw1_len && j < netw2_len {

            let n1: &Node<f32> = &genome_1[netw1_len - 1 - i];
            let n2: &Node<f32> = &genome_2[netw2_len - 1 - j];

            if n1.gin == n2.gin {
                // println!("Same GIN {:#?}", n1);
                arn.push(n1.clone());

                i += 1;
                j += 1;

            } else {

                if !n1_gin_vector.contains(&n2.gin) {
                    while i < netw1_len && j < netw2_len - 1 {//}&& !common_gin_in_both_vector.contains(&n2.gin) {
                        let n2: &Node<f32> = &genome_2[netw2_len - 1 - j];

                        if !n1_gin_vector.contains(&n2.gin) {
                            arn.push(
                                Node::new_nan(n2.gin, n2.iota)
                            );
                        } else {
                            break;
                        }
                        j += 1;

                    }
                }

                let n2: &Node<f32> = &genome_2[netw2_len - 1 - j];
                if n1.gin == n2.gin {
                    continue;
                }

                if n2_gin_vector.contains(&n1.gin) {
                    // println!("{:#?}", n1);
                    // arn.push(n1.clone());
                    arn.push(Node::new_nan(n1.gin, n1.iota));
                    j += 1;
                } else {
                    // println!("Push NaN: {:#?}", n1);
                    arn.push(n1.clone());
                    // arn.push(Node::new_nan(n1.gin));
                }
                
                i += 1;

            }
        }


        arn.reverse();
        arn
    }


    /// Sort arn 2 to match the arn 1 order.
    fn sort_arn(arn_1: &[Node<f32>], arn_2: &[Node<f32>]) -> Vec<Node<f32>> {
        
        assert_eq!(arn_1.len(), arn_2.len(), "arn_1.len() != arn_2.len()");
        let arn_len: usize = arn_1.len();

        let mut arn_sorted: Vec<Node<f32>> = Vec::with_capacity(arn_len);

        // HashMap of the indices of the correspondinf GIN inside arn_2: HashMap<GIN, Index>.
        let mut arn_map: HashMap<usize, usize> = HashMap::with_capacity(arn_len);
        for i in 0..arn_len {
            arn_map.insert(arn_2[i].gin, i);
        }

        for i in 0..arn_len {
            let node_index_of_gin: usize = *arn_map.get(&arn_1[i].gin).expect("Fail to lookup arn GIN index.");
            arn_sorted.push(arn_2[node_index_of_gin].clone());
        }

        arn_sorted
    }


    /// Update the iota values of all the Neuron Nodes inside an ARN.
    fn arn_update_iota(arn_1: &[Node<f32>], arn_2: &[Node<f32>]) -> Vec<Node<f32>> {
        // use std::collections::VecDeque;

        let arn_len: usize = arn_1.len();
        // let mut arn_updated: Vec<Node<f32>> = Vec::with_capacity(arn_len);
        let mut arn_updated: Vec<Node<f32>> = arn_1.to_vec().clone();

        // HashMap of the indices of the correspondinf GIN inside arn_2: HashMap<GIN, Index>.
        let mut arn_map: HashMap<usize, usize> = HashMap::with_capacity(arn_len);
        for i in 0..arn_len {
            arn_map.insert(arn_2[i].gin, i);
        }

        // let mut max_gin: usize = *arn_1.iter().map(|n| n.gin).collect::<Vec<usize>>().iter().max().unwrap_or(&1_usize);

        // max_gin += 1;
        // let nan = Node::new_nan(max_gin, 1);

        let mut stack_1: Vec<Node<f32>> = Vec::with_capacity(arn_len);
        let mut stack_2: Vec<Node<f32>> = Vec::with_capacity(arn_len);

        let debug: bool = false;

        // let mut i: usize = 0;
        for i in 0..arn_len {
            let n1: Node<f32> = arn_1[arn_len - 1 - i].clone();
            let n2: Node<f32> = arn_2[arn_len - 1 - i].clone();

                {

                if n1.allele == Allele::Neuron && n2.allele == Allele::Neuron {
                    if debug {
                        println!("\n\n");
                        println!("\n\nif n1 == n2 == Neuron");
                    }

                    stack_1 = stack_1.iter().filter(|n| n.allele != Allele::NaN).map(|m| m.clone()).collect();
                    stack_2 = stack_2.iter().filter(|n| n.allele != Allele::NaN).map(|m| m.clone()).collect();

                    if debug {
                        println!("Stack 1:");
                        Network::pretty_print(&stack_1);
                        println!("Stack 2:");
                        Network::pretty_print(&stack_2);
                        println!();
                    }
                    
                    let mut stack_tmp_1: Vec<Node<f32>> = Vec::with_capacity(stack_1.len());
                    let slice_len_1: usize = (1 - n1.iota) as usize;

                    let mut j: usize = 0;
                    while j < slice_len_1 && !stack_1.is_empty() {

                        if stack_1.last().unwrap().allele == Allele::NaN {
                            stack_1.pop().unwrap();
                        } else {
                            stack_tmp_1.push(stack_1.pop().unwrap());
                            j += 1;
                        }
                    }
                    stack_1 = stack_1.iter().filter(|n| n.allele != Allele::NaN).map(|m| m.clone()).collect();

                    let mut stack_tmp_2: Vec<Node<f32>> = Vec::with_capacity(stack_2.len());
                    let slice_len_2: usize = (1 - n2.iota) as usize;

                    let mut j: usize = 0;
                    while j < slice_len_2 && !stack_2.is_empty() {

                        if stack_2.last().unwrap().allele == Allele::NaN {
                            stack_2.pop().unwrap();
                        } else {
                            stack_tmp_2.push(stack_2.pop().unwrap());
                            j += 1;
                        }
                    }
                    stack_2 = stack_2.iter().filter(|n| n.allele != Allele::NaN).map(|m| m.clone()).collect();


                    if debug {
                        println!("Stack 1:");
                        Network::pretty_print(&stack_1);
                        println!("Stack tmp 1:");
                        Network::pretty_print(&stack_tmp_1);

                        println!("Stack 2:");
                        Network::pretty_print(&stack_2);
                        println!("Stack tmp 2:");
                        Network::pretty_print(&stack_tmp_2);
                    }

                    let vnr1: Vec<&Node<f32>> = stack_tmp_1.iter().map(|n| n).collect::<Vec<&Node<f32>>>();
                    let vnr2: Vec<&Node<f32>> = stack_tmp_2.iter().map(|n| n).collect::<Vec<&Node<f32>>>();
                    let common_input_number: i32 = Network::count_common_input(&vnr1, &vnr2);

                    let in_1: i32 = 1 - n1.iota;
                    let in_2: i32 = 1 - n2.iota;
                    arn_updated[arn_len - 1 - i].iota = 1 - ( in_1 + in_2 - common_input_number );

                    if debug {
                        println!("iota = 1 - ( {} + {} - {} ) = {}", in_1, in_2, common_input_number, arn_updated[arn_len - 1 - i].iota);
                        println!("\n\n");
                    }
 
                    // max_gin += 1;
                    let dummy_input = Node::new(Allele::Input, 0, n1.gin, 0.0, IOTA_INPUT_VALUE, INPUT_NODE_DEPTH_VALUE);
                    stack_1.push(dummy_input.clone());
                    stack_2.push(dummy_input.clone());

                } else if n1.allele == Allele::Neuron {
                    if debug {
                        println!("Condition Neuron 1 {:#?}", n1);

                        stack_1 = stack_1.iter().filter(|n| n.allele != Allele::NaN).map(|m| m.clone()).collect();
                        println!("\n\nif n1 == Allele::Neuron");
                        println!("Stack 1:");
                        Network::pretty_print(&stack_1);
                        println!("Stack 2:");
                        Network::pretty_print(&stack_2);
                        println!();
                    }

                    let mut stack_tmp_1: Vec<Node<f32>> = Vec::with_capacity(stack_1.len());
                    let slice_len_1: usize = (1 - n1.iota) as usize;

                    let mut j: usize = 0;
                    while j < slice_len_1 && !stack_1.is_empty() {

                        if stack_1.last().unwrap().allele == Allele::NaN {
                            stack_1.pop().unwrap();
                        } else {
                            stack_tmp_1.push(stack_1.pop().unwrap());
                            j += 1;
                        }
                    }
                    stack_1 = stack_1.iter().filter(|n| n.allele != Allele::NaN).map(|m| m.clone()).collect();

                    // max_gin += 1;
                    // let dummy_input = Node::new(Allele::Input, 0, max_gin, 0.0, IOTA_INPUT_VALUE, INPUT_NODE_DEPTH_VALUE);
                    let dummy_input = Node::new(Allele::Input, 0, n1.gin, 0.0, IOTA_INPUT_VALUE, INPUT_NODE_DEPTH_VALUE);
                    stack_1.push(dummy_input.clone());

                    if debug {
                        println!("Stack 1:");
                        Network::pretty_print(&stack_1);
                        println!("Stack tmp 1:");
                        Network::pretty_print(&stack_tmp_1);
                        println!("Stack 2:");
                        Network::pretty_print(&stack_2);
                    }

                } else if n2.allele == Allele::Neuron {
                    if debug {
                        println!("Condition Neuron 2 {:#?}", n2);
                    }

                    stack_2 = stack_2.iter().filter(|n| n.allele != Allele::NaN).map(|m| m.clone()).collect();

                    if debug {
                        println!("\n\nif n2 == Allele::Neuron");
                        println!("Stack 1:");
                        Network::pretty_print(&stack_1);
                        println!("Stack 2:");
                        Network::pretty_print(&stack_2);
                        println!();
                    }

                    let mut stack_tmp_2: Vec<Node<f32>> = Vec::with_capacity(stack_2.len());
                    let slice_len_2: usize = (1 - n2.iota) as usize;

                    let mut j: usize = 0;
                    while j < slice_len_2 && !stack_2.is_empty() {

                        if stack_2.last().unwrap().allele == Allele::NaN {
                            stack_2.pop().unwrap();
                        } else {
                            stack_tmp_2.push(stack_2.pop().unwrap());
                            j += 1;
                        }
                    }
                    stack_2 = stack_2.iter().filter(|n| n.allele != Allele::NaN).map(|m| m.clone()).collect();

                    // max_gin += 1;
                    // let dummy_input = Node::new(Allele::Input, 0, max_gin, 0.0, IOTA_INPUT_VALUE, INPUT_NODE_DEPTH_VALUE);
                    let dummy_input = Node::new(Allele::Input, 0, n2.gin, 0.0, IOTA_INPUT_VALUE, INPUT_NODE_DEPTH_VALUE);
                    stack_2.push(dummy_input.clone());

                    if debug {
                        println!("Stack 1:");
                        Network::pretty_print(&stack_1);
                        println!("Stack 2:");
                        Network::pretty_print(&stack_2);
                        println!("Stack tmp 2:");
                        Network::pretty_print(&stack_tmp_2);
                    }

                } else {
                    stack_1.push(n1);
                    stack_2.push(n2);
                }
            }

        }
        arn_updated
    }


    /// Counts the number of inputs to a subnetwork.
    fn count_common_input(arn_1: &[&Node<f32>], arn_2: &[&Node<f32>]) -> i32 {
        let mut accu: i32 = 0;

        let gin_v1: Vec<usize> = arn_1.iter().map(|n| n.gin).collect();
        let gin_v2: Vec<usize> = arn_2.iter().map(|n| n.gin).collect();

        for gin in gin_v1.iter() {
            if gin_v2.contains(&gin) {
                accu += 1;
            }
        }

        accu
    }


    /// Render an articial neural network to a dot file for a better visualization purpose.
    /// cf.: https://www.graphviz.org/documentation/.
    pub fn render_to_dot(
        &self,
        file_name: &str,
        graph_name: &str,
        print_weight: bool,
    ) -> ::std::io::Result<()> {

        use utils::create_parent_directory;
        use std::fs::File;
        use std::io::BufWriter;


        create_parent_directory(file_name)?;
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

            // Print roof.
            print!("+");
            for _ in 0..genome_chunk.len() {
                print!("{:^9}+", format!("{:-^9}", ""));
            }
            println!("");

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
                print!("{:^9}|", format!("{:^3}({})", node.iota, 1 - node.iota));
            }
            println!("");

            // Print floor.
            print!("+");
            for _ in 0..genome_chunk.len() {
                print!("{:^9}+", format!("{:-^9}", ""));
            }
            println!("");

            // Print number of inputs.
            // print!("|");
            // for node in genome_chunk.iter() {
            //     print!("{:^9}|", format!("i#{:^3}", 1 - node.iota));
            // }
            // println!("");

        }
    }
}
