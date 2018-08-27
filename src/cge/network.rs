//! A flexible encoding method enables one to design an efficient evolutionary
//! method that can evolve both the structures and weights of neural networks.
//! The genome in EANT2 is designed by taking this fact into consideration.
//!
//! A genome in EANT2 is a linear genome consisting of genes (nodes) that can take different forms (alleles).

use activation::TransferFunctionTrait;
use cge::node::Allele::*;
use cge::node::{Node, INPUT_NODE_DEPTH_VALUE, IOTA_INPUT_VALUE};
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::io::Write;


/// The representation of an Artificial Neural Network (ANN) using the Common Genetic Encoding
/// (CGE) (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.8729).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Network<T> {
    // The linear genome is represented by a vector of Node.
    pub genome: Vec<Node<T>>,
    // This let us map the values of all Input Node. Used during the evaluation phase.
    // Those values are processed by the ReLu transfert function during evaluation time.
    pub input_map: Vec<T>,
    // Neuron value processed by a `Transfer Function`.
    pub neuron_map: Vec<T>,
    // Neuron index lookup table: <genome[i].id, index in self.genome>
    pub neuron_indices_map: HashMap<usize, usize>,
    // The number of Output in this Network. It's a constant value as well.
    pub output_size: usize,
}


impl Network<f32> {
    /// Generating the initial linear genome use the grow method by default.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Network::new_simple(input_size, output_size)
    }


    /// Starting from simple initial structures is the way it is done by nature and most of the
    /// other evolutionary methods.
    pub fn new_simple(input_size: usize, output_size: usize) -> Self {
        let input_map: Vec<f32> = vec![0.0_f32; input_size];
        let mut genome: Vec<Node<f32>> = Vec::new();

        // Global Innovation Number is used to keep track of the Nodes to enable crossover
        // later during evolution.
        let mut gin: usize = 1;

        let iota_for_each_neuron: i32 = 1 - input_size as i32;
        for output in 0..output_size {
            let mut input_node_vector: Vec<Node<f32>> =
                Network::gen_input_node_vector(gin, &input_map);

            let neuron: Node<f32> = Node::new(
                Neuron { id: output },
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

        let neuron_map: Vec<f32> = vec![0.0_f32; output_size];
        let neuron_indices_map: HashMap<usize, usize> = Network::compute_neuron_indices(&genome);

        Network {
            genome,
            input_map,
            neuron_map,
            neuron_indices_map,
            output_size,
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
    pub fn new_grow(input_vec: &Vec<f32>, output_size: usize) -> Self {
        let genome: Vec<Node<f32>> = Vec::new();
        let input_map: Vec<f32> = input_vec.clone();
        let neuron_map: Vec<f32> = vec![];
        let neuron_indices_map: HashMap<usize, usize> = Network::compute_neuron_indices(&genome);

        Network {
            genome,
            input_map,
            neuron_map,
            neuron_indices_map,
            output_size,
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
    pub fn new_full(input_vec: &Vec<f32>, output_size: usize) -> Self {
        let genome: Vec<Node<f32>> = Vec::new();
        let input_map: Vec<f32> = input_vec.iter().map(|i| i.relu()).collect();
        let neuron_map: Vec<f32> = vec![];
        let neuron_indices_map: HashMap<usize, usize> = Network::compute_neuron_indices(&genome);

        Network {
            genome,
            input_map,
            neuron_map,
            neuron_indices_map,
            output_size,
        }
    }


    /// Builds and returns the genome from the research papers we use to implement EANT2.
    pub fn build_from_example() -> Self {
        let input_map = vec![1_f32, 1_f32];
        let neuron_map: Vec<f32> = vec![0.0; 4];
        let genome: Vec<Node<f32>> = vec![
            Node::new(Neuron { id: 0 }, 1, 0.6, -1, 0),
            Node::new(Neuron { id: 1 }, 2, 0.8, -1, 1),
            Node::new(Neuron { id: 3 }, 3, 0.9, -1, 2),
            Node::new(
                Input { label: 0 },
                4,
                0.1,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Input { label: 1 },
                5,
                0.4,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Input { label: 1 },
                6,
                0.5,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(Neuron { id: 2 }, 7, 0.2, -3, 1),
            Node::new(JumpForward { source_id: 3 }, 8, 0.3, IOTA_INPUT_VALUE, 2),
            Node::new(
                Input { label: 0 },
                9,
                0.7,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Input { label: 1 },
                10,
                0.8,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(JumpRecurrent { source_id: 0 }, 11, 0.2, IOTA_INPUT_VALUE, 2),
        ];
        let neuron_indices_map: HashMap<usize, usize> = Network::compute_neuron_indices(&genome);

        Network {
            genome,
            input_map,
            neuron_map,
            neuron_indices_map,
            output_size: 1,
        }
    }


    /// Builds and returns the parent 1's genome from the research papers we use to implement EANT2.
    pub fn _build_parent1_from_example() -> Self {
        let input_map = vec![1_f32, 1_f32];
        let neuron_map: Vec<f32> = vec![0.0; 4];
        let genome: Vec<Node<f32>> = vec![
            Node::new(Neuron { id: 0 }, 1, 0.6, -1, 0),
            Node::new(Neuron { id: 1 }, 2, 0.8, -1, 1),
            Node::new(Neuron { id: 3 }, 7, 0.9, -1, 2),
            Node::new(
                Input { label: 0 },
                8,
                0.1,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Input { label: 1 },
                9,
                0.4,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Input { label: 1 },
                3,
                0.5,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(Neuron { id: 2 }, 4, 0.2, -3, 1),
            Node::new(JumpForward { source_id: 3 }, 13, 0.3, IOTA_INPUT_VALUE, 2),
            Node::new(
                Input { label: 0 },
                5,
                0.7,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Input { label: 1 },
                6,
                0.8,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(JumpRecurrent { source_id: 0 }, 12, 0.2, IOTA_INPUT_VALUE, 2),
        ];
        let neuron_indices_map: HashMap<usize, usize> = Network::compute_neuron_indices(&genome);

        Network {
            genome,
            input_map,
            neuron_map,
            neuron_indices_map,
            output_size: 1,
        }
    }


    /// Builds and returns the parent 2's genome from the research papers we use to implement EANT2.
    pub fn _build_parent2_from_example() -> Self {
        let input_map = vec![1_f32, 1_f32];
        let neuron_map: Vec<f32> = vec![0.0; 3];
        let genome: Vec<Node<f32>> = vec![
            Node::new(Neuron { id: 0 }, 1, 0.8, -1, 0),
            Node::new(Neuron { id: 1 }, 2, 1.0, -2, 1),
            Node::new(JumpRecurrent { source_id: 1 }, 10, 0.3, IOTA_INPUT_VALUE, 1),
            Node::new(
                Input { label: 0 },
                11,
                0.1,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Input { label: 1 },
                3,
                0.7,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(Neuron { id: 2 }, 4, 0.9, -1, 1),
            Node::new(
                Input { label: 0 },
                5,
                0.5,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
            Node::new(
                Input { label: 1 },
                6,
                2.8,
                IOTA_INPUT_VALUE,
                INPUT_NODE_DEPTH_VALUE,
            ),
        ];
        let neuron_indices_map: HashMap<usize, usize> = Network::compute_neuron_indices(&genome);

        Network {
            genome,
            input_map,
            neuron_map,
            neuron_indices_map,
            output_size: 1,
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

        let mut input_node_vec: Vec<Node<f32>> =
            Network::gen_input_node_vector(gin + 1, &input_map);
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
        let neuron: Node<f32> = Node::new(Neuron { id: neuron_id }, gin, 0.0, iota, depth + 1);
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
                Input { label: i },
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
                Neuron { id } => {
                    if node.depth > depth_source && id != source_id {
                        indices_v.push(id);
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
        let mut is_forward: bool;
        let mut potential_target_neuron_indices: Vec<usize>;

        match thread_rng().gen_range(0_usize, 2_usize) {
            0 => {
                is_forward = true;
                potential_target_neuron_indices =
                    self.get_potential_neuron_indices_for_jf_node(source_id, depth_source);
            }
            _ => {
                is_forward = false;
                potential_target_neuron_indices =
                    self.neuron_indices_map.keys().map(|x| *x).collect();
            }
        };

        // If there is no possibility to form a new forward connection in the current genome, we
        // fall back to a recurrent connection, which should be in the most cases possible.
        if is_forward && potential_target_neuron_indices.len() < 1 {
            is_forward = false;
            potential_target_neuron_indices = self.neuron_indices_map.keys().map(|x| *x).collect();
        }

        // Still, we check if there is some possible solution.
        if potential_target_neuron_indices.len() > 0 {
            let source_id: usize = *thread_rng()
                .choose(&potential_target_neuron_indices)
                .expect("Fail to draw a jumper connection id to link to an existing Neuron.");
            let jumper = if is_forward {
                JumpForward { source_id }
            } else {
                JumpRecurrent { source_id }
            };
            let jumper: Node<f32> = Node::new(
                jumper,
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
    fn compute_neuron_indices(genome: &[Node<f32>]) -> HashMap<usize, usize> {
        // The initial capacity of the HashMap is totally arbitrary.
        let mut neuron_indices_hashmap: HashMap<usize, usize> =
            HashMap::with_capacity(genome.len() / 2 as usize);
        for i in 0..genome.len() {
            if let Neuron { id } = genome[i].allele {
                neuron_indices_hashmap.insert(id, i);
            }
        }
        neuron_indices_hashmap.shrink_to_fit();
        neuron_indices_hashmap
    }


    /// Compute the indices of each Neuron gin into a HashMap lookup table.
    pub fn compute_neurons_gin_indices_map(genome: &[Node<f32>]) -> HashMap<usize, usize> {
        // The initial capacity of the HashMap is totally arbitrary.
        let mut neurons_gin_indices_hashmap: HashMap<usize, usize> =
            HashMap::with_capacity(genome.len() / 2 as usize);

        for i in 0..genome.len() {
            if let Neuron { id } = genome[i].allele {
                neurons_gin_indices_hashmap.insert(genome[i].gin, i);
            }
        }
        neurons_gin_indices_hashmap.shrink_to_fit();
        neurons_gin_indices_hashmap
    }


    /// Build a HashMap of reference to Neuron Node. HashMap <gin, &Node>.
    fn build_gin_neuron_map<'a>(genome: &'a [Node<f32>]) -> HashMap<usize, &Node<f32>> {
        // The initial capacity of the HashMap is totally arbitrary.
        let mut neurons_gin_indices_hashmap: HashMap<usize, &Node<f32>> =
            HashMap::with_capacity(genome.len() / 2 as usize);

        for node in genome {
            if let Neuron { .. } = node.allele {
                neurons_gin_indices_hashmap.insert(node.gin, node);
            }
        }
        neurons_gin_indices_hashmap.shrink_to_fit();
        neurons_gin_indices_hashmap
    }


    /// Helper function to return the sorted Neuron's gin values of a Network.
    fn get_neuron_gin_vector(genome: &[Node<f32>]) -> Vec<usize> {
        let mut v: Vec<usize> = Network::compute_neurons_gin_indices_map(genome)
            .iter()
            .map(|(gin, _)| *gin)
            .collect();
        v.sort();
        v
    }


    /// Update a Network.
    pub fn update(&mut self) {
        self.update_network_attributes();
    }


    /// Update the attributes of the network.
    pub fn update_network_attributes(&mut self) {
        self.neuron_indices_map = Network::compute_neuron_indices(&self.genome);

        let neuron_id_max: usize = *self
            .genome
            .iter()
            .filter_map(|n| {
                if let Neuron { id } = n.allele {
                    Some(id)
                } else {
                    None
                }
            }).collect::<Vec<usize>>()
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

    /// Evaluate a genome, using the values from `self.inputs`, and the current values of each
    /// `Allele::Neuron` of  `self.genome`.
    /// 
    /// # Return
    ///
    /// The values associated to each outputs, in a Vec, in the same order as the outputs.
    ///
    pub fn evaluate(&mut self) -> Option<Vec<f32>> {
        // This function evaluates the genome like a stack machine, for right to left.
        //
        // The evaluation of each `Node`s mostly follows the order of `self.genome` (in reverse),
        // unless it meets `JumpForward` allele, in this case the algorithm will
        // evalute the sub-genome needed to get the output of the source neuron of the `JumpForward` node,
        // and then will continue its execution.
        
        // this stack represents the data-flow of the algorithm
        let mut neuron_input_stack: Vec<f32> = Vec::new();
        // map any neuron ID to its computed value.
        // this map serves 2 purposes:
        // * it tracks the IDs of each neuron processed in this evaluation
        // * it allows a quick lookup for it
        let mut neuron_id_to_value: HashMap<usize, f32> = HashMap::new();
        // build a lookup table: Neuron ID to neuron index in self.genome
        let mut neuron_id_to_idx: HashMap<usize, usize> = HashMap::new();
        for (i, node) in self.genome.iter().enumerate() {
            if let Neuron { id } = node.allele {
                neuron_id_to_idx.insert(id, i);
            }
        }
        // this stack keeps the node indices to compute,
        // it's updated during the evaluation of each node,
        // especially when evaluation a `JumpForward` allele
        let mut nodes_indices_to_process: Vec<usize> = (0..self.genome.len()).collect();
        // store the neuron ids we want to evaluate for JumpForwards
        // this helps us track if we were evaluating a subnetwork for a JF or not.
        let mut forward_in_process: Vec<usize> = Vec::new();
        'node_eval: while let Some(node_idx) = nodes_indices_to_process.pop() {
            let node = &self.genome[node_idx];
            match &node.allele {
                Input { label } => { 
                    let input_value = self.input_map[*label].relu() * node.w;
                    neuron_input_stack.push(input_value);
                },
                JumpRecurrent { source_id } => {
                    // fetch the output value of the neuron
                    // * If it's already been compute in this evaluation process, take that
                    // evaluation
                    //  * Otherwise take the inherent `.value` of the node (from the last
                    //  evaluation call)
                    let source_node_idx = *neuron_id_to_idx.get(&source_id)?;
                    let source_neuron_value = neuron_id_to_value
                        .get(&source_id)
                        .unwrap_or(&self.genome[source_node_idx].value);
                    let value = node.w * source_neuron_value;
                    neuron_input_stack.push(value);
                },
                JumpForward { source_id } => {
                    // fetch the ouput of the neuron source,
                    // if it's has already been evaluated, use it,
                    // otherwise first evaluate it by:
                    //  - moving the stack machine to the subnetwork needed to compute the neuron output
                    //  - re-try to evaluate this node
                    match neuron_id_to_value.get(&source_id) {
                        None => {
                            // un-pop the current node, because we can not evaluate it right now
                            nodes_indices_to_process.push(node_idx);
                            // figure out wich node to evaluate before evaluating the current node.
                            let mut sub_network_indices: Vec<usize> = Vec::new();
                            let mut iota: i32 = 0;
                            let mut idx = *neuron_id_to_idx.get(&source_id)?;
                            while iota != 1 && idx < self.genome.len() {
                                iota += self.genome[idx].iota;
                                sub_network_indices.push(idx);
                                idx += 1;
                            }
                            // append them to the node to evaluate list
                            nodes_indices_to_process.extend(sub_network_indices.iter());
                            // keep track of the current forward process
                            forward_in_process.push(*source_id);
                            continue 'node_eval;
                        },
                        // If we already computed the value of this neuron in this evaluation call
                        Some(neuron_value) => {
                            // To get here, either:
                            // * the neuron was already evalutated way before this step,
                            // * or we just evaluated it for this node
                            // If we did, we must pop out first stack value, as we are not interested 
                            // in the sub-network output, by in the value of the root neuron of it.
                            let mut must_pop_stack = false;
                            if let Some(neuron_id) = forward_in_process.last() {
                                must_pop_stack = neuron_id == source_id;
                            }
                            if must_pop_stack {
                                let _ = neuron_input_stack.pop()?;
                                let _ = forward_in_process.pop()?;
                            }
                            // now we can process it
                            let mut value = node.w * neuron_value.isrlu(0.1);
                            neuron_input_stack.push(value);
                        },
                    }
                },
                Neuron { id } => {
                    // Compute the output of this neuron, and update its value
                    let mut neuron_value: f32 = 0.;
                    let input_len = 1 - node.iota;
                    for _ in 0..input_len {
                        neuron_value += neuron_input_stack.pop()?;
                    }
                    neuron_value = neuron_value.relu();
                    // stores it for jumping connection
                    neuron_id_to_value.insert(*id, neuron_value);
                    // store it as an input to the rest ro the genome
                    neuron_input_stack.push(neuron_value * node.w);
                },
                _Nan => {},
            }
        }
        // update the values of each neurons
        for node in self.genome.iter_mut() {
            if let Neuron { id } = node.allele {
                let value = *neuron_id_to_value.get(&id)?;
                node.value = value;
            }
        }
        Some(neuron_input_stack)
    }


    /// Returns if a Network is considered valid.
    pub fn is_valid(&mut self) -> bool {
        let inputs: Vec<f32> = vec![1.0; self.input_map.len()];

        self.update();

        let iota_sum: i32 = self
            .genome
            .iter()
            .map(|n| n.iota)
            .collect::<Vec<i32>>()
            .iter()
            .sum();
        if self.output_size as i32 != iota_sum {
            println!(
                "\n>> iota_sum {} != {} self.output_size",
                iota_sum, self.output_size
            );
            println!("Test subject :");
            Network::pretty_print(&self.genome);
            // panic!("iota_sum {} != {} self.output_size", iota_sum, self.output_size);
            return false;
        }

        self.update_input(&inputs);
        let output: Vec<f32> = self.evaluate().unwrap_or(vec![]);
        // let output: Vec<f32> = self.evaluate().unwrap_or(return false);
        // let output: Vec<f32> = self.evaluate().unwrap();

        if output.len() != self.output_size {
            println!(
                "output.len() {} != {} self.output_size",
                output.len(),
                self.output_size
            );
            return false;
        }

        true
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
                Neuron { .. } => {
                    iota += node.iota;
                }
                _ => {
                    iota += 1;
                }
            }

            output_vec.push(node);

            if i >= input_len {
                println!(
                    "@build_jf_slice:\n\t>> Genome end reached. Looking for N{} at index {}, but we reached index {} and nothing.",
                    neuron_id, neuron_index, i
                );
                Network::pretty_print(input_vec);
                panic!("@build_jf_slice:");
                break;
            }
            i += 1;
        }

        output_vec.shrink_to_fit();
        output_vec
    }


    /// Build a vector of GIN value from a slice.
    pub fn build_ref_input_subnetwork<'a>(genome: &[&'a Node<f32>]) -> Vec<&'a Node<f32>> {
        let genome_len: usize = genome.len();
        let mut inputs: Vec<&Node<f32>> = Vec::with_capacity(genome_len);

        let mut iota: i32 = genome[0].iota;
        let mut i: usize = 1;


        while iota != 1 && i < genome_len {
            let node: &Node<f32> = &genome[i];

            match node.allele {
                Neuron { .. } => {
                    let mut iota_tmp = node.iota;
                    inputs.push(node);

                    i += 1;

                    while iota_tmp != 1 && i < genome_len {
                        let node: &Node<f32> = &genome[i];
                        iota_tmp += node.iota;
                        i += 1;
                    }
                    iota += 1;
                }
                _ => {
                    i += 1;
                    iota += node.iota;
                    inputs.push(node);
                }
            }
        }

        inputs.dedup_by(|a, b| a.gin.eq(&b.gin));
        inputs
    }


    /// Third and last refactoring of the crossover operator !
    pub fn crossover(
        network_1: &Network<f32>,
        network_2: &Network<f32>,
        fitness_1: f32,
        fitness_2: f32,
        debug: bool,
    ) -> Network<f32> {
        // 1° Build each inputs sub-network into a HashMap and use a Vector to store order
        // information.
        // 2° Starting from the end, build each new and updated input sub-network corresponding to
        // the merge of the common original sub-networks (and remove duplicate during the process).
        // 3° Update Neuron's iotas using the sub-networks of inputs from (1).

        let mut offspring = network_1.clone();

        // Let's create sub-networks of each Neuron in each Network.
        let (n1_order, n1_isn_hm) = Network::build_subnetworks_lookup_table(network_1);
        let (n2_order, n2_isn_hm) = Network::build_subnetworks_lookup_table(network_2);

        if debug {
            println!("\nn1_order = {:?}", n1_order);
            println!("n1_isn_hm =");
            Network::_pp_subnetworks_lookup_table(&n1_order, &n1_isn_hm);


            println!("\nn2_order = {:?}", n2_order);
            println!("n2_isn_hm =");
            Network::_pp_subnetworks_lookup_table(&n2_order, &n2_isn_hm);
        }

        // At this point we are going to recombine each sub-network and store them into a loopup
        // table (HashMap).
        let isn_merged_hm =
            Network::merge_and_update_subnetworks(&n1_order, &n2_order, &n1_isn_hm, &n2_isn_hm);

        if debug {
            println!("isn_merged_hm ( 1 ):");
            Network::_pp_subnetworks_lookup_table(&n1_order, &isn_merged_hm);

            println!("isn_merged_hm ( 2 ):");
            Network::_pp_subnetworks_lookup_table(&n2_order, &isn_merged_hm);
        }

        // Now we need to recombine all those sub-networks into one linear genome.
        // To do so, we will start from the output Nodes and unravel their inputs.

        // So first we need to get the GIN of the output Nodes (and because every individual of
        // a population started from the same structure, this array of value should be the same for
        // our two networks.
        // Also, we need those array to have the info about the order of the Neurons in the initial genomes.
        let n1_output_gin_vector: Vec<usize> =
            Network::get_neuron_gin_vector(&network_1.genome)[..network_1.output_size].to_vec();
        let n2_output_gin_vector: Vec<usize> =
            Network::get_neuron_gin_vector(&network_2.genome)[..network_2.output_size].to_vec();
        // So we check if our previous assomption is correct. You know, just in case ^^'
        assert_eq!(
            n1_output_gin_vector, n2_output_gin_vector,
            "Inconsistency between Network output Nodes."
        );

        let n1_gin_neuron_map: HashMap<usize, &Node<f32>> =
            Network::build_gin_neuron_map(&network_1.genome);
        let n2_gin_neuron_map: HashMap<usize, &Node<f32>> =
            Network::build_gin_neuron_map(&network_2.genome);

        let mut gin_neuron_map: HashMap<usize, &Node<f32>> =
            HashMap::with_capacity(n1_gin_neuron_map.len());
        for (k, v) in n1_gin_neuron_map.iter().chain(n2_gin_neuron_map.iter()) {
            gin_neuron_map.insert(*k, v);
        }

        if debug {
            println!("n1_output_gin_vector = {:?}", n1_output_gin_vector);
            println!("n2_output_gin_vector = {:?}", n2_output_gin_vector);
            println!("n1_gin_neuron_map = {:?}", n1_gin_neuron_map.keys());
            println!("n2_gin_neuron_map = {:?}", n2_gin_neuron_map.keys());
        }

        // At last we can recombine them.
        offspring.genome =
            Network::recombine_subnetworks(&n1_output_gin_vector, &gin_neuron_map, &isn_merged_hm);

        // Finally, we need to update each Node's weight in order to priorotize the fitest ones.
        offspring.update_weights(&network_1.genome, &network_2.genome, fitness_1, fitness_2);

        offspring.update();
        offspring
    }


    /// Build a sub-network of inputs in a lookup table (HashMap) and store the order in a Vector.
    pub fn build_subnetworks_lookup_table(
        network: &Network<f32>,
    ) -> (Vec<usize>, HashMap<usize, Vec<&Node<f32>>>) {
        let neuron_order: Vec<usize> = network
            .genome
            .iter()
            .filter_map(|n| {
                if let Neuron { .. } = n.allele {
                    Some(n.gin)
                } else {
                    None
                }
            }).collect();

        let genome_len: usize = network.genome.len();
        let mut inputs_subnetwork_hm: HashMap<usize, Vec<&Node<f32>>> =
            HashMap::with_capacity(neuron_order.len());

        for i in 0..genome_len {
            let node = &network.genome[i];

            if let Neuron { .. } = node.allele {
                let slice: Vec<&Node<f32>> = network.genome[i..].iter().map(|n| n).collect();
                let subnetwork = Network::build_ref_input_subnetwork(&slice);

                inputs_subnetwork_hm.insert(node.gin, subnetwork);
            }
        }

        (neuron_order, inputs_subnetwork_hm)
    }


    pub fn _pp_subnetworks_lookup_table(n_order: &[usize], hm: &HashMap<usize, Vec<&Node<f32>>>) {
        for i in n_order {
            let sub_net = hm.get(i).unwrap();
            println!("NGin ({:^3}) :", i);
            Network::_pretty_print_refs(&sub_net);
        }
    }


    /// Build new subnetwork.
    fn merge_and_update_subnetworks<'a>(
        n1_order: &[usize],
        n2_order: &[usize],
        n1_subnet_hm: &HashMap<usize, Vec<&'a Node<f32>>>,
        n2_subnet_hm: &HashMap<usize, Vec<&'a Node<f32>>>,
    ) -> HashMap<usize, Vec<&'a Node<f32>>> {
        let mut hm_merged: HashMap<usize, Vec<&Node<f32>>> = HashMap::new();

        let mut common_neuron_gin: Vec<usize> = Vec::with_capacity(n1_order.len() + n2_order.len());
        common_neuron_gin.append(&mut n1_order.to_vec());
        common_neuron_gin.append(&mut n2_order.to_vec());
        common_neuron_gin.sort();
        common_neuron_gin.dedup();
        // println!("\n>> common_neuron_gin = {:?}\n", common_neuron_gin);

        // First we concentrate on the leaves of the trees.
        {
            for (gin, inputs) in n1_subnet_hm {
                if !common_neuron_gin.contains(gin) {
                    hm_merged.insert(*gin, inputs.clone());
                }
            }
            for (gin, inputs) in n2_subnet_hm {
                if !common_neuron_gin.contains(gin) {
                    hm_merged.insert(*gin, inputs.clone());
                }
            }
        }


        // Next we focus on each vertex.
        {
            // Let's iterate on the first network.
            for i in 0..n1_order.len() {
                let node_gin: usize = n1_order[n1_order.len() - 1 - i];

                if common_neuron_gin.contains(&node_gin) {
                    let _default: Vec<&Node<f32>> = vec![];
                    let sub_net_1 = n1_subnet_hm.get(&node_gin).unwrap_or(&_default);

                    let _default: Vec<&Node<f32>> = vec![];
                    let sub_net_2 = n2_subnet_hm.get(&node_gin).unwrap_or(&_default);

                    let merged_sub_net = Network::merge_input_subnetworks(&sub_net_1, &sub_net_2);

                    hm_merged.insert(node_gin, merged_sub_net);
                }
            }

            // And then on the second one.
            for i in 0..n2_order.len() {
                let node_gin: usize = n2_order[n2_order.len() - 1 - i];

                // We want to grab the left over Neurons and avoid to iterate on the ones we
                // already process during the last loop.
                if common_neuron_gin.contains(&node_gin) && !hm_merged.contains_key(&node_gin) {
                    let _default: Vec<&Node<f32>> = vec![];
                    let sub_net_1 = n1_subnet_hm.get(&node_gin).unwrap_or(&_default);

                    let _default: Vec<&Node<f32>> = vec![];
                    let sub_net_2 = n2_subnet_hm.get(&node_gin).unwrap_or(&_default);

                    let merged_sub_net = Network::merge_input_subnetworks(&sub_net_1, &sub_net_2);

                    hm_merged.insert(node_gin, merged_sub_net);
                }
            }
        }

        hm_merged
    }


    fn merge_input_subnetworks<'a>(
        sub_net_1: &[&'a Node<f32>],
        sub_net_2: &[&'a Node<f32>],
    ) -> Vec<&'a Node<f32>> {
        // Fist we need to separate the inputs and neurons Node of each sub-network.
        // So we gather all the I, JF and JR on one side and all the N in the other side.
        let (sub_net_of_input_1, sub_net_of_neuron_1) =
            Network::separate_input_and_neuron_from_a_subnetwork(sub_net_1);
        let (mut sub_net_of_input_2, mut sub_net_of_neuron_2) =
            Network::separate_input_and_neuron_from_a_subnetwork(sub_net_2);

        // And we merge those two sub-networks of input Node into one.
        let mut sub_net_of_input_merged: Vec<&Node<f32>> = sub_net_of_input_1;
        sub_net_of_input_merged.append(&mut sub_net_of_input_2);

        // And then we remove the duplicates.
        sub_net_of_input_merged.sort_by(|a, b| a.gin.cmp(&b.gin));
        sub_net_of_input_merged.dedup_by(|a, b| a.gin.eq(&b.gin));

        // Next we need to focus on the Neuron Nodes.
        let mut sub_net_of_neuron_merged: Vec<&Node<f32>> = sub_net_of_neuron_1;
        sub_net_of_neuron_merged.append(&mut sub_net_of_neuron_2);

        // And then once again, we remove the duplicates.
        sub_net_of_neuron_merged.sort_by(|a, b| a.gin.cmp(&b.gin));
        sub_net_of_neuron_merged.dedup_by(|a, b| a.gin.eq(&b.gin));

        let mut merged_sub_net = sub_net_of_input_merged;
        merged_sub_net.append(&mut sub_net_of_neuron_merged);

        merged_sub_net
    }


    /// Separate input Node (I, JF and JR) and Neuron Node in two Vectors.
    fn separate_input_and_neuron_from_a_subnetwork<'a>(
        sub_net: &[&'a Node<f32>],
    ) -> (Vec<&'a Node<f32>>, Vec<&'a Node<f32>>) {
        // On one side we gather all the input Nodes: I, JF and JR.
        let input_nodes: Vec<&Node<f32>> = sub_net
            .iter()
            .filter_map(|n| {
                if let Neuron { .. } = n.allele {
                    None
                } else {
                    Some(*n)
                }
            }).collect();

        // And on the other side we gather only the Neuron ones.
        let neuron_nodes: Vec<&Node<f32>> = sub_net
            .iter()
            .filter_map(|n| {
                if let Neuron { .. } = n.allele {
                    Some(*n)
                } else {
                    None
                }
            }).collect();

        (input_nodes, neuron_nodes)
    }


    /// Recombine the sub-networks into a linear genome.
    fn recombine_subnetworks<'a>(
        output_node_gin_vector: &[usize],
        gin_neuron_map: &HashMap<usize, &'a Node<f32>>,
        merged_sub_net_hm: &HashMap<usize, Vec<&'a Node<f32>>>,
    ) -> Vec<Node<f32>> {
        let mut recombined_genome: Vec<Node<f32>> = Vec::new();

        for gin in output_node_gin_vector {
            // println!("Looking for gin {}", gin);
            let _output_neuron_ref: &Node<f32> = gin_neuron_map.get(&gin).unwrap();
            let mut output_neuron: Node<f32> = _output_neuron_ref.clone();

            let _neurons_input_ref: Vec<&Node<f32>> = merged_sub_net_hm.get(&gin).unwrap().to_vec();
            let mut neurons_inputs: Vec<Node<f32>> =
                _neurons_input_ref.iter().map(|n| (*n).clone()).collect();

            // We update the iota of each Neuron while passing by...
            output_neuron.iota = 1 - neurons_inputs.len() as i32;
            recombined_genome.push(output_neuron);

            for input_node in neurons_inputs {
                // println!("\tInput_node {}", input_node.gin);
                if let Neuron { .. } = input_node.allele {
                    let mut input_neuron_node = input_node.clone();
                    let mut recombined_sub_genome = Network::recombine_subnetworks(
                        &[input_neuron_node.gin],
                        gin_neuron_map,
                        merged_sub_net_hm,
                    );

                    input_neuron_node.iota = 1 - recombined_sub_genome.len() as i32;
                    // recombined_genome.push(input_neuron_node);
                    recombined_genome.append(&mut recombined_sub_genome);
                } else {
                    recombined_genome.push(input_node);
                }
            }
        }

        recombined_genome
    }


    /// Update the weights of each Node of the linear genome by choosing the fitest ones from their
    /// parents.
    fn update_weights(
        &mut self,
        genome_1: &[Node<f32>],
        genome_2: &[Node<f32>],
        fitness_1: f32,
        fitness_2: f32,
    ) {
        let mut n1_gin_node_lookup_table: HashMap<usize, &Node<f32>> =
            HashMap::with_capacity(genome_1.len());
        let mut n2_gin_node_lookup_table: HashMap<usize, &Node<f32>> =
            HashMap::with_capacity(genome_2.len());

        // Build up some lookup table to find each Nodes.
        for n in genome_1 {
            n1_gin_node_lookup_table.insert(n.gin, &n);
        }
        for n in genome_2 {
            n2_gin_node_lookup_table.insert(n.gin, &n);
        }

        for mut node in &mut self.genome {
            let w1: f32 = match n1_gin_node_lookup_table.get(&node.gin) {
                Some(n) => n.w,
                None => 0.0,
            };
            let w2: f32 = match n2_gin_node_lookup_table.get(&node.gin) {
                Some(n) => n.w,
                None => 0.0,
            };

            if fitness_1 >= fitness_2 {
                if w1 > 0.0 {
                    node.w = w1;
                } else {
                    node.w = w2;
                }
            } else {
                if w1 > 0.0 {
                    node.w = w1;
                } else {
                    node.w = w2;
                }
            }
        }
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
        use utils::create_parent_directory;


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

                for i in 0..self.output_size {
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
                    if let JumpForward { source_id } = node.allele {
                        empty = false;
                        let msg: String = format!("JF{} ", source_id);
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
                    if let JumpRecurrent { source_id } = node.allele {
                        empty = false;
                        let msg: String = format!("JR{} ", source_id);
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
                                JumpRecurrent { source_id } => format!("JR{} ", source_id),
                                JumpForward { source_id } => format!("JF{} ", source_id),
                                Neuron { id } => format!("N{} ", id),
                                Input { label } => format!("I{} ", label),
                                _ => format!("NaN"),
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
                    Input { label } => {
                        stack.push(format!("I{label}", label = label));
                        if print_weight {
                            stack.push(format!("[label=\"{w:.3}\"]", w = node.w));
                        } else {
                            stack.push(format!("[label=\"\"]"));
                        }
                    }
                    JumpForward { source_id } => {
                        let _msg: String =
                            format!("    {t}{i}[label=\"{t}{i}\"];\n", t = "JF", i = source_id);
                        // writer.write(_msg.as_bytes())?;

                        stack.push(format!("JF{id}", id = source_id));
                        if print_weight {
                            stack.push(format!("[label=\"{w:.3}\"]", w = source_id));
                        } else {
                            stack.push(format!("[label=\"\"]"));
                        }
                    }
                    JumpRecurrent { source_id } => {
                        let _msg: String =
                            format!("    {t}{i}[label=\"{t}{i}\"];\n", t = "JR", i = source_id);
                        // writer.write(_msg.as_bytes())?;

                        stack.push(format!("JR{id}", id = source_id));
                        if print_weight {
                            stack.push(format!("[label=\"{w:.3}\"]", w = node.w));
                        } else {
                            stack.push(format!("[label=\"\"]"));
                        }
                    }
                    Neuron { id } => {
                        let neuron_input_number: usize = (1 - node.iota) as usize;
                        // println!("Dot {} {:#?}", neuron_input_number, node);

                        for _ in 0..neuron_input_number {
                            let msg: String = format!(
                                "    {x} -> N{id}{label};\n",
                                id = id,
                                label = stack.pop().expect("No more label in stack."),
                                x = stack.pop().expect("Empty stack."),
                            );

                            writer.write(msg.as_bytes())?;
                        }

                        stack.push(format!("N{id}", id = id));
                        if print_weight {
                            stack.push(format!("[label=\"{w:.3}\"]", w = node.w));
                        } else {
                            stack.push(format!("[label=\"\"]"));
                        }
                    }
                    NaN => {
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
                    Input { label } => print!("{:^9}|", format!(" I{:<3}", label)),
                    Neuron { id } => print!("{:^9}|", format!(" N{:<3}", id)),
                    JumpForward { source_id } => print!("{:^9}|", format!(" JF{:<3}", source_id)),
                    JumpRecurrent { source_id } => print!("{:^9}|", format!(" JR{:<3}", source_id)),
                    NaN => print!("{:^9}|", format!(" X{:<3}", 'x')),
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

    /// Pretty print the liear genome on a line.
    pub fn _pretty_print_refs(genome: &[&Node<f32>]) {
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
                    Input { label } => print!("{:^9}|", format!(" I{:<3}", label)),
                    Neuron { id } => print!("{:^9}|", format!(" N{:<3}", id)),
                    JumpForward { source_id } => print!("{:^9}|", format!(" JF{:<3}", source_id)),
                    JumpRecurrent { source_id } => print!("{:^9}|", format!(" JR{:<3}", source_id)),
                    NaN => print!("{:^9}|", format!(" X{:<3}", 'x')),
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
        }
    }
}
