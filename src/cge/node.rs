//! A genome in EANT2 is a linear genome consisting of genes (nodes) that can take different forms (alleles).

use activation::TransferFunctionTrait;

/// This value is used to check the completeness of the linear genome and sub-linear genomes
/// (sub-Networks).
pub const IOTA_INPUT_VALUE: i32 = 1;


/// The forms that can be taken by a gene can either be a neuron, or an input to the neural network, or
/// a jumper connecting two neurons.
/// The jumper genes are introduced by structural mutation along the evolution path.
/// This enum is used during the 'evaluation' process when we compute the output of the artificial
/// neural network without decoding it.
#[derive(Clone, Debug, PartialEq)]
pub enum Allele {
    // A Neuron is the basic unit process of an artificial neural network.
    Neuron,
    // An Input gene (Node) encodes an input to the network (for example, a sensory signal).
    // Each Input corresponds to the inputs fed to our artificial neural network.
    Input,
    //  A jumper gene encoding a forward connection represents a connection starting from a neuron at
    //  a higher depth and ending at a neuron at a lower depth. The depth of a neuron node in a linear
    //  genome is the minimal number of neuron nodes that must be traversed to get from the output
    //  neuron to the neuron node, where the output neuron and the neuron node
    //  lie within the same sub-network that starts from the output neuron.
    JumpForward,
    //  On the other hand, a jumper gene encoding a recurrent connection represents a connection
    //  between neurons having the same depth, or a connection starting
    //  from a neuron at a lower depth and ending at a neuron at a higher depth.
    JumpRecurrent,
}

/// A flexible encoding method enables one to design an efficient evolutionary method that can evolve both
/// the structures and weights of neural networks. The genome in EANT is designed by taking this fact 
/// into consideration.
/// A genome in EANT is a linear genome consisting of genes (nodes) that can take different forms (alleles), 
/// symbolized by the [`Allele`] enum.
#[derive(Clone, Debug, PartialEq)]
pub struct Node<T> {
    // This is the specific form a gene can take. Possible values are contained in 
    // the 'Allele' enum: Neuron, Input, JumpForward, JumpRecurrent.
    pub allele: Allele,
    // Unique global identification number. This is used especially by jumper connections.
    pub id: usize,
    // The weight encodes the synaptic strength of the connection between the Node
    // coded by the gene and the Neuron to which it is connected.
    pub w: T,
    // Number of inputs of the Neuron.
    pub iota: i32,
    // Stores the result of its current computation. This is useful since the results of signals
    // at recurrent links are available at the next time step.
    pub value: T,
}

impl Node<f32> {
    pub fn new(allele: Allele, id: usize, w: f32, iota: i32) -> Self {
        Node {
            allele,
            id,
            w,
            iota,
            value: 0_f32,
        }
    }
}


impl TransferFunctionTrait<f32> for Node<f32> {
    fn isrlu(&self, alpha: f32) -> f32 {
        if self.value < 0_f32 {
            self.value / (1_f32 + alpha * (self.value * self.value)).sqrt()
        } else {
            self.value
        }
    }

    fn isru(&self, alpha: f32) -> f32 {
        self.value / (1.0 + alpha * (self.value * self.value)).sqrt()
    }

    fn relu(&self) -> f32 {
        if self.value < 0.0 { 0.0 }
        else { self.value }
    }

}

























//
//
// /// The forms that can be taken by a gene can either be a neuron, or an input to the neural network, or
// /// a jumper connecting two neurons.
// /// The jumper genes are introduced by structural mutation along the evolution path.
// #[derive(Clone, Debug, PartialEq)]
// pub enum Allele {
//     Node(Node),
//     Neuron(Neuron),
//     Input(Input),
//     JumpForward(JumpForward),
//     JumpRecurrent(JumpRecurrent),
// }
//
//
// /// 
// #[derive(Clone, Debug, PartialEq)]
// pub struct Neuron {
//     // Unique global identification number. This is used especially by jumper connections.
//     id: usize,
//     // The weight encodes the synaptic strength of the connection between the Node
//     // coded by the gene and the Neuron to which it is connected.
//     w: f32,
//     // Number of inputs of the Neuron.
//     iota: i32,
//     // Stores the result of its current computation. This is useful since the results of signals
//     // at recurrent links are available at the next time step.
//     value: f32,
// }
//
// impl Neuron {
//     pub fn new(id: usize, w: f32, iota: i32) -> Self {
//         Neuron {
//             id,
//             w,
//             iota,
//             value: 0_f32,
//         }
//     }
// }
//
//
// /// An Input gene (Node) encodes an input to the network (for example, a sensory signal).
// /// Each Input corresponds to the inputs fed to our artificial neural network.
// #[derive(Clone, Debug, PartialEq)]
// pub struct Input {
//     // Unique global identification number.
//     id: usize,
//     // The weight encodes the synaptic strength of the connection between the Node
//     // coded by the gene and the Neuron to which it is connected.
//     w: f32,
//     // Number of inputs of the Neuron. For Input node it's a constant value set to 1.
//     iota: i32,
//     // Stores the result of its current computation. This is useful since the results of signals
//     // at recurrent links are available at the next time step.
//     value: f32,
// }
//
// impl Input {
//     pub fn new(id: usize, w: f32, value: f32) -> Self {
//         Input {
//             id,
//             w,
//             iota: IOTA_INPUT_VALUE,
//             value,
//         }
//     }
// }
//
//
// ///  A jumper gene encoding a forward connection represents a connection starting from a neuron at
// ///  a higher depth and ending at a neuron at a lower depth. The depth of a neuron node in a linear
// ///  genome is the minimal number of neuron nodes that must be traversed to get from the output
// ///  neuron to the neuron node, where the output neuron and the neuron node
// ///  lie within the same sub-network that starts from the output neuron.
// #[derive(Clone, Debug, PartialEq)]
// pub struct JumpForward {
//     // This ID of the Neuron this jumper symbolizes.
//     id: usize,
//     // The weight encodes the synaptic strength of the connection between the Node
//     // coded by the gene and the Neuron to which it is connected.
//     w: f32,
//     // Number of inputs of the Neuron. For Input node it's a constant value set to 1.
//     iota: i32,
//     // I'm not sure about what this value should be at the moment tbh ^^'
//     value: f32,
// }
//
// impl JumpForward {
//     pub fn new(id: usize, w: f32) -> Self {
//         JumpForward {
//             id,
//             w,
//             iota: IOTA_INPUT_VALUE,
//             value: 0.0,
//         }
//     }
// }
//
//
// ///  On the other hand, a jumper gene encoding a recurrent connection represents a connection
// ///  between neurons having the same depth, or a connection starting
// ///  from a neuron at a lower depth and ending at a neuron at a higher depth.
// #[derive(Clone, Debug, PartialEq)]
// pub struct JumpRecurrent {
//     // This ID of the Neuron this jumper symbolizes.
//     id: usize,
//     // The weight encodes the synaptic strength of the connection between the Node
//     // coded by the gene and the Neuron to which it is connected.
//     w: f32,
//     // Number of inputs of the Neuron. For Input node it's a constant value set to 1.
//     iota: i32,
//     // This value is init at 0.0 and will be gathered from the value of the Neuron
//     // this recurrent jumper connection symbolizes.
//     value: f32,
// }
//
// impl JumpRecurrent {
//     pub fn new(id: usize, w: f32) -> Self {
//         JumpRecurrent {
//             id,
//             w,
//             iota: IOTA_INPUT_VALUE,
//             value: 0.0,
//         }
//     }
// }
