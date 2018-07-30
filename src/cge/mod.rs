//! EANT2 represents neural networks and their parameters in a compact way, the Compact Genetic Encoding (cge) aka the "linear genome".
//! It encodes the topology of the network implicitly by the order of its elements (genes).
//!
//! In CGE, a genotype consists of a string of genes, whose order implicitly represents the
//! topology of a network. Each gene takes on a specific form (allele): it can either be a vertex
//! gene (Neuron), an input gene, or a jumper gene. A vertex gene encodes a vertex of a network,
//! an input gene encodes an input to the network (for example, a sensory signal), and
//! a jumper gene encodes a connection between two vertices.

pub mod network;
pub mod node;
