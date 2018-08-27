//! A genome in EANT2 is a linear genome consisting of genes (nodes) that can take different forms (alleles).

use activation::TransferFunctionTrait;
use rand::{thread_rng, Rng};
// use serde_derive::Serialize;


/// This value is used to check the completeness of the linear genome and sub-linear genomes
/// (sub-Networks).
pub const IOTA_INPUT_VALUE: i32 = 1;
/// Default value of an Input Node.
pub const INPUT_NODE_DEPTH_VALUE: u16 = 999;


/// This enum describes the forms that can be taken by a gene.
///
/// A gene can either be:
///  * a neuron,
///  * an input to the neural network,
///  * a jumper connecting two neurons.
///
/// The jumper genes are introduced by structural mutation along the evolution path.
/// This enum is used during the 'evaluation' process when we compute the output of the artificial
/// neural network without decoding it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Allele {
    // A Neuron is the basic unit process of an artificial neural network.
    Neuron { id: usize },
    // An Input gene (Node) encodes an input to the network (for example, a sensory signal).
    // Each Input corresponds to the inputs fed to our artificial neural network.
    Input { label: usize },
    //  A jumper gene encoding a forward connection represents a connection starting from a neuron at
    //  a higher depth and ending at a neuron at a lower depth. The depth of a neuron node in a linear
    //  genome is the minimal number of neuron nodes that must be traversed to get from the output
    //  neuron to the neuron node, where the output neuron and the neuron node
    //  lie within the same sub-network that starts from the output neuron.
    JumpForward { source_id: usize },
    //  On the other hand, a jumper gene encoding a recurrent connection represents a connection
    //  between neurons having the same depth, or a connection starting
    //  from a neuron at a lower depth and ending at a neuron at a higher depth.
    JumpRecurrent { source_id: usize },
    // Not a Node special kind of Allele, used during crossover operation to align the common parts
    // of two genomes.
    NaN,
}


/// A flexible encoding method enables one to design an efficient evolutionary method that can evolve both
/// the structures and weights of neural networks. The genome in EANT is designed by taking this fact
/// into consideration.
///
/// A genome in EANT is a linear genome consisting of genes (nodes) that can take different forms (alleles),
/// symbolized by the [`Allele`] enum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node<T> {
    // This is the specific form a gene can take. Possible values are contained in
    // the 'Allele' enum: Neuron, Input, JumpForward, JumpRecurrent.
    pub allele: Allele,
    // Global Innovation Number represents a chronology of every gene in the system.
    pub gin: usize,
    // The weight encodes the synaptic strength of the connection between the Node
    // coded by the gene and the Neuron to which it is connected.
    // w ∈ R.
    pub w: T,
    // Step size of a parametric mutation or learning rate.
    pub sigma: T,
    // Integer symbolizing the number of inputs of the Neuron and defined by the following
    // equation: 1 - n, where n is the actual number of input of the Neuron.
    pub iota: i32,
    // Stores the result of its current computation. This is useful since the results of signals
    // at recurrent links are available at the next time step.
    pub value: T,
    // The depth of a Node. Needed for JF and JR connections addition mutation.
    pub depth: u16,
}


impl Node<f32> {
    pub fn new(allele: Allele, gin: usize, w: f32, iota: i32, depth: u16) -> Self {
        use genetic_algorithm::individual::LEARNING_RATE_THRESHOLD;
        Node {
            allele,
            gin,
            w,
            sigma: LEARNING_RATE_THRESHOLD as f32,
            iota,
            value: 0_f32,
            depth,
        }
    }


    /// Returns a special kind of allele: NaN (Not a Node).
    pub fn new_nan(gin: usize, iota: i32) -> Self {
        Node {
            allele: Allele::NaN,
            gin,
            w: 0.0,
            sigma: 0.0,
            iota,
            value: 0.0,
            depth: 0,
        }
    }

    /// Returns a proper random weight in the space: [0.0, 1.0], with one decimal value.
    pub fn random_weight() -> f32 {
        thread_rng().gen_range(0_f32, 1_f32)
    }

    /// Returns either or not self.allele is a Neuron
    pub fn is_neuron(&self) -> bool {
        if let Allele::Neuron { .. } = self.allele {
            true
        } else {
            false
        }
    }
}


 use std::fmt;
 impl fmt::Display for Node<f32> {
     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
         let msg: String = match &self.allele {
             Allele::Input { label } => format!("I{:<3}", label),
             Allele::Neuron { id } => format!("N{:<3}", id),
             Allele::JumpForward { source_id } => format!("JF{:<3}", source_id),
             Allele::JumpRecurrent { source_id } => format!("JR{:<3}", source_id),
             Allele::NaN => format!("X{:<3}", 'x'),
         };
         write!(f, "{:^3}", &msg)
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
        if self.value < 0.0 {
            0.0
        } else {
            self.value
        }
    }

    fn sigmoids(&self) -> f32 {
        1.0 / (1.0 + (-self.value).exp())
    }
}
