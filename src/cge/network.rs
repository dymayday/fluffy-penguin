//! Copy-Pasta some info about our Artificial Neural Network here buddy.
//! A flexible encoding method enables one to design an efficient evolutionary
//! method that can evolve both the structures and weights of neural networks. 
//! The genome in EANT2 is designed by taking this fact into consideration.
//!
//! A genome in EANT2 is a linear genome consisting of genes (nodes) that can take different forms (alleles).

use std::collections::HashMap;
use cge::node::{Allele, Node, IOTA_INPUT_VALUE};
use activation::TransferFunctionTrait;

/// A Network.
#[derive(Clone, Debug, PartialEq)]
pub struct Network<T> {
    // The linear genome is represented by a vector of Node.
    pub genome: Vec<Node<T>>,
    // Shadowing the genome is the only workaround to fix some immutable with mutable borrow issue.
    // The sizes of the networks should not be an issue though.
    shadow_genome: Vec<Node<T>>,
    // This let us map the values of all Input Node. Used during the evaluation phase.
    // Those values are processed by the ReLu transfert function during evaluation time.
    input_map: Vec<T>,
    // Neuron value processed by a `Transfer Function`.
    neuron_map: Vec<T>,
    // Neuron index lookup table.
    neuron_indices_map: HashMap<usize, usize>,
    // Number of Input in this Network. It has to be a constant value.
    iota_number: i32,
    // The number of Output in this Network. It's a constant value as well.
    omega_number: i32,
}

impl Network<f32> {
    pub fn new(input_vec: &Vec<f32>, omega_number: i32) -> Self {
        Network::new_full(input_vec, omega_number)
    }

    /// Builds and returns a Network from a list of input value using the `grow` method.
    /// For a given maximum depth, the grow method produces linear genomes encoding neural networks
    /// of irregular shape because a node is assigned to a randomly generated neuron node having a
    /// random number of inputs or to a randomly selected input node.
    ///
    /// There is two important thing to notice here:
    /// * The order of the inputs is particularly important.
    /// * The weights of each input are randomly attributed.
    pub fn new_grow(input_vec: &Vec<f32>, omega_number: i32) -> Self {
        // We gather the number of input from the size of the input vector.
        let iota_number: i32 = input_vec.len() as i32;
        // And determine the approximate size of the vectors the network will use.
        let max_vector_size: usize = (iota_number * omega_number) as usize;

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
            iota_number,
            omega_number,
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
    pub fn new_full(input_vec: &Vec<f32>, omega_number: i32) -> Self {
        // We gather the number of input from the size of the input vector.
        let iota_number: i32 = input_vec.len() as i32;
        // And determine the approximate size of the vectors the network will use.
        let max_vector_size: usize = (iota_number * omega_number) as usize;

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
            iota_number,
            omega_number,
        }

    }

    /// Builds and returns the genome from the research papers we use to implement EANT2.
    pub fn build_from_example() -> Self {
        let input_map = vec![1_f32, 1_f32, ];
        let neuron_map: Vec<f32> = vec![0.0; 4];
        let genome: Vec<Node<f32>> = vec![
            Node::new(Allele::Neuron, 0, 0.6, -1),
            Node::new(Allele::Neuron, 1, 0.8, -1),
            Node::new(Allele::Neuron, 3, 0.9, -1),
            Node::new(Allele::Input, 0, 0.1, IOTA_INPUT_VALUE),
            Node::new(Allele::Input, 1, 0.4, IOTA_INPUT_VALUE),
            Node::new(Allele::Input, 1, 0.5, IOTA_INPUT_VALUE),
            Node::new(Allele::Neuron, 2, 0.2, -3),
            Node::new(Allele::JumpForward, 3, 0.3, IOTA_INPUT_VALUE),
            Node::new(Allele::Input, 0, 0.7, IOTA_INPUT_VALUE),
            Node::new(Allele::Input, 1, 0.8, IOTA_INPUT_VALUE),
            Node::new(Allele::JumpRecurrent, 0, 0.2, IOTA_INPUT_VALUE),
        ];
        let shadow_genome = genome.clone();
        let neuron_indices_map: HashMap<usize, usize> = Network::compute_neuron_indices(&genome);

        Network {
            genome,
            shadow_genome,
            input_map, 
            neuron_map,
            neuron_indices_map,
            iota_number: 2,
            omega_number: 1,
        }
    }


    /// Compute the indexes of the Neuron in the linear genome so we can find them easily during
    /// the evaluation process. This function is meant to be call only once at init time.
    fn compute_neuron_indices(genome: &Vec<Node<f32>>) -> HashMap<usize, usize> {
        let mut neuron_indices_hashmap: HashMap<usize, usize> = HashMap::with_capacity(genome.len());
        for i in 0..genome.len() {
            if genome[i].allele == Allele::Neuron {
                neuron_indices_hashmap.insert(genome[i].id, i);
            }
        }
        neuron_indices_hashmap.shrink_to_fit();
        neuron_indices_hashmap
    }



    /// Evaluate the linear genome to compute the output of the artificial neural network without decoding it.
    pub fn evaluate(&mut self) -> Vec<f32> {
        // println!("neuron_map: {:?}", self.neuron_map);
        // println!("neuron_indices_map: {:#?}", self.neuron_indices_map);
        let g = self.genome.clone();
        self.evaluate_slice(&g)
    }


    /// Evaluate a sub-linear genome to compute the output of an artificial neural sub-network without decoding it.
    fn evaluate_slice(&mut self, input: &[Node<f32>]) -> Vec<f32> {
        let mut stack: Vec<f32> = Vec::with_capacity(input.len());
        
        let input_len: usize = input.len();
        // println!("input_len = {}", input_len);

        for i in 0..input_len {
            let mut node: Node<f32> = input[input_len - i - 1].clone();
            // println!("\n{:#?}", node);

            match node.allele {
                Allele::Input => {
                    stack.push(self.input_map[node.id].relu() * node.w);
                },
                Allele::Neuron => {
                    let neuron_input_number: usize = (1-node.iota) as usize;
                    let mut neuron_output: f32 = 0.0;
                    for _ in 0..neuron_input_number {
                        // neuron_input_number += stack.pop().unwrap_or(0.0_f32);
                        // [TODO]: Remove this expect for an unwrap_or maybe ?
                        neuron_output += stack.pop().expect("The evaluate stack is empty.");
                    }
                    
                    node.value = neuron_output;
                    let neuron_index: usize = *self.neuron_indices_map.get(&node.id).expect(
                            &format!("Fail to lookup the node id = {}", node.id)
                        );
                    self.genome[neuron_index].value = neuron_output;
                    let activated_neuron_value: f32 = node.isrlu(0.1);
                    // Update the neuron value in the neuron_map with its activated value from its
                    // transfert function to be used by jumper connection nodes.
                    self.neuron_map[node.id] = activated_neuron_value;

                    stack.push(activated_neuron_value * node.w);
                },
                Allele::JumpRecurrent => {
                    let recurrent_neuron_value: f32 = self.neuron_map[node.id];
                    stack.push(recurrent_neuron_value * node.w);
                },
                Allele::JumpForward => {
                    // We need to evaluate a slice of our linear genome in a different depth.
                    let forwarded_node_index: usize = *self.neuron_indices_map.get(&node.id).expect(
                            &format!("Fail to lookup the node id = {}", node.id)
                        );


                    let forwarded_node: Node<f32> = self.shadow_genome[forwarded_node_index].clone();

                    let sub_genome_slice_length: usize = (1 - forwarded_node.iota) as usize;
                    let mut sub_genome_slice: Vec<Node<f32>> = Vec::with_capacity(sub_genome_slice_length);
                    
                    sub_genome_slice.push(forwarded_node);

                    for sub_genome_index in 0..sub_genome_slice_length {
                        sub_genome_slice.push(
                                self.shadow_genome[forwarded_node_index + sub_genome_index + 1].clone()
                            );
                    }

                    // println!("\n\nEval slice: {:#?}", sub_genome_slice);
                    stack.append(&mut self.evaluate_slice(&sub_genome_slice));
                }
                // _ => println!("Unknown Allele encountered: {:#?}", node)
            }
            // println!("Stack: [{:>2}] = {:?} ", i, stack);
            
            // let mut input = String::new();
            // ::std::io::stdin().read_line(&mut input)
            //     .ok()
            //     .expect("Couldn't read line");    
                
        }

        assert_eq!(
            stack.len(), self.omega_number as usize,
            "Evaluated genome output length differt from expected output length: {} != {}",
            stack.len(), self.omega_number
            );
        stack

    }
}
