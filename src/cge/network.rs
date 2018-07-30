//! Copy-Pasta some info about our Artificial Neural Network here buddy.
//! A flexible encoding method enables one to design an efficient evolutionary
//! method that can evolve both the structures and weights of neural networks. 
//! The genome in EANT2 is designed by taking this fact into consideration.
//!
//! A genome in EANT2 is a linear genome consisting of genes (nodes) that can take different forms (alleles).

use rand::{thread_rng, Rng};
use cge::node::{Allele, Node, IOTA_INPUT_VALUE};

/// A Network.
#[derive(Clone, Debug, PartialEq)]
pub struct Network<T> {
    // The linear genome is represented by a vector of Node.
    pub genome: Vec<Node<T>>,
    // This let us map the values of all Input Node. Used during the evaluation phase.
    input_map: Vec<T>,
    // Number of Input in this Network. It has to be a constant value.
    iota_number: i32,
    // The number of Output in this Network. It's a constant value as well.
    omega_number: i32,
}

impl Network<f32> {
    pub fn new(iota_number: i32, omega_number: i32) -> Self {
        let genome: Vec<Node<f32>> = Vec::with_capacity((iota_number + omega_number) as usize);

        Network {
            genome,
            input_map: vec![],
            iota_number,
            omega_number,
        }
    }

    /// Builds and returns a Network from a list of input value.
    /// There is two important thing to notice here:
    /// * The order of the inputs is particularly important.
    /// * The weights of each input are randomly attributed.
    pub fn new_from_imput_vec(input_vec: Vec<f32>, omega_number: i32) -> Self {
        let iota_number: i32 = input_vec.len() as i32;
        let genome: Vec<Node<f32>> = Vec::with_capacity((iota_number + omega_number) as usize);

        let _w = thread_rng().gen_range(0.0_f32, 1.0_f32);

        Network {
            genome,
            input_map: input_vec.clone(),
            iota_number,
            omega_number,
        }

    }

    /// Builds and returns the genome from the research papers we use to implement EANT2.
    pub fn build_from_example() -> Self {
        let input_map = vec![1_f32, 1_f32, ];
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

        Network {
            genome,
            input_map,
            iota_number: 2,
            omega_number: 1,
        }
    }
}
