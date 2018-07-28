//! A genome in EANT is a linear genome consisting of genes (nodes) that can take different forms (alleles).

/// The forms that can be taken by a gene can either be a neuron, or an input to the neural network, or 
/// a jumper connecting two neurons.
pub enum Allele {
    Neuron,
    Input,
}


#[derive(Clone, Debug, PartialEq)]
pub struct Neuron {
    // Unique global identification number. This is especially used by jumper connections.
    id: usize,
    // The weight encodes the synaptic strength of the connection between the Node
    // coded by the gene and the Neuron to which it is connected.
    w: f32,
    // Number of inputs of the Neuron.
    inputs: i32,
    // Stores the result of its current computation. This is useful since the results of signals
    // at recurrent links are available at the next time step.
    res: f32,
}

impl Neuron {
    pub fn new(id: usize, w: f32, inputs: i32, ) -> Self {
        Neuron {
            id,
            w,
            inputs,
            res: 0_f32,
        }
    }
}


#[derive(Clone, Debug, PartialEq)]
pub struct Input {
    // 
    // The weight encodes the synaptic strength of the connection between the Node
    // coded by the gene and the Neuron to which it is connected.
    w: f32,
    // Stores the result of its current computation. This is useful since the results of signals
    // at recurrent links are available at the next time step.
    res: f32,
}
