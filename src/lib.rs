//! This is the core of our A.I. engine. "EANT2 with prune" is the algorithm we choose to implement and some
//! documentation can be found in this repository.
//! We choose EANT2 because of its capability to evolve its own structure and prune the connections
//! that are unnecessary. I find it very cool and powerful and I invit you to learn more about this
//! beautiful peace of art =].

extern crate rand;
extern crate rayon;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate bincode;

pub mod activation;
pub mod cge;
pub mod genetic_algorithm;
pub mod utils;


#[cfg(test)]
mod activation_tests {
    use activation;

    #[test]
    fn activation_test_relu() {
        assert_eq!(0.5703423, activation::relu(0.5703423_f32));
        assert_eq!(0_f32, activation::relu(-10.5703423_f32));
    }


    #[test]
    fn activation_test_sigmoids() {
        assert_eq!(0.7310586_f32, activation::sigmoids(1_f32));
    }

}

#[cfg(test)]
mod specimen {
    use genetic_algorithm::individual::Specimen;

    #[test]
    fn structural_mutation() {
        let mut specimen_origin: Specimen<f32> = Specimen::new_from_example();
        let mut specimen_mutated: Specimen<f32> = specimen_origin.clone();
        let input_vector: Vec<f32> = vec![1_f32; 2];

        specimen_origin.update_input(&input_vector);
        let specimen_origin_output: Vec<f32> = specimen_origin.evaluate();

        specimen_mutated.update_input(&input_vector);

        // Get a not accurate but valid GIN value from the size of the Specimen's genome.
        let mut gin: usize = specimen_origin.ann.genome.len() * 2;
        let mut nn_id: usize = specimen_origin.ann.genome.len();
        for _ in 0..30 {
            let (gin_tmp, nn_id_tmp) = specimen_mutated
                .structural_mutation(0.5, gin, nn_id)
                .unwrap();
            gin = gin_tmp;
            nn_id = nn_id_tmp;
            specimen_mutated.update_input(&input_vector);
            let nodes_str: Vec<String> = specimen_mutated.ann.genome.iter()
                .map(|n| format!("{}{:^2}", &n, &n.iota))
                .collect();
            println!("{}", nodes_str.join(" ")); 
            let iota_sum: i32 = specimen_mutated.ann.genome.iter().map(|n| n.iota).sum();
            println!("{}", iota_sum);

            assert_eq!(
                specimen_origin_output.len(),
                specimen_mutated.evaluate().len()
            );
        }
    }
}


#[cfg(test)]
mod network {
    use cge::Network;

    #[test]
    fn evaluation_ann_from_example() {
        let mut network = Network::build_from_example();
        // test first pass
        assert_eq!(
            network.evaluate(),
            Some(vec![0.65400004_f32])
        );
        // test second pass
        assert_eq!(
            network.evaluate(),
            Some(vec![0.68016005])
        );
    }

    #[test]
    /// This tests the evaluation of a network which contains
    /// A JF node right after the its source Neuron.
    fn evaluation_jump_forward_after_Neuron() {
        use cge::{Allele::*, Node, INPUT_NODE_DEPTH_VALUE, IOTA_INPUT_VALUE};
        use std::collections::HashMap;
        let genome: Vec<Node<f32>> = vec![
            Node { allele: Neuron { id: 0 }, gin: 1, w: 0.6, sigma: 0.01, iota: -2, value: 0.0, depth: 0 },
            Node { allele: JumpRecurrent { source_id: 2 }, gin: 23, w: 0.0, sigma: 0.01, iota: 1, value: 0.0, depth: 1 },
            Node { allele: Neuron { id: 1 }, gin: 2, w: 0.8, sigma: 0.01, iota: -2, value: 0.0, depth: 1 },
            Node { allele: JumpForward { source_id: 3 }, gin: 24, w: 0.0, sigma: 0.01, iota: 1, value: 0.0, depth: 2 },
            Node { allele: Neuron { id: 3 }, gin: 3, w: 0.9, sigma: 0.01, iota: 0, value: 0.0, depth: 2 },
            Node { allele: Input { label: 1 }, gin: 5, w: 0.4, sigma: 0.01, iota: 1, value: 0.0, depth: 999 },
            Node { allele: Input { label: 1 }, gin: 6, w: 0.5, sigma: 0.01, iota: 1, value: 0.0, depth: 999 },
            Node { allele: Neuron { id: 2 }, gin: 7, w: 0.2, sigma: 0.01, iota: -3, value: 0.0, depth: 1 },
            Node { allele: JumpForward { source_id: 3 }, gin: 8, w: 0.3, sigma: 0.01, iota: 1, value: 0.0, depth: 2 },
            Node { allele: Input { label: 0 }, gin: 9, w: 0.7, sigma: 0.01, iota: 1, value: 0.0, depth: 999 },
            Node { allele: Input { label: 1 }, gin: 10, w: 0.8, sigma: 0.01, iota: 1, value: 0.0, depth: 999 },
            Node { allele: JumpRecurrent { source_id: 0 }, gin: 11, w: 0.2, sigma: 0.01, iota: 1, value: 0.0, depth: 2 },
        ];

        let mut net = Network {
            genome: genome,
            input_map: vec![1., 1.],
            neuron_map: vec![0., 0., 0., 0.],
            neuron_indices_map: HashMap::new(),
            output_size: 1,
        };
        let output = net.evaluate().unwrap();
        assert_eq!(output.len(), 1)        
    }
}


#[cfg(test)]
mod node_tests {
    use cge::{Allele, Node};

    #[test]
    fn neuron() {
        Node::new(Allele::Neuron { id: 0 }, 0_usize, 0.3_f32, 1_i32, 0);
    }

    #[test]
    fn input() {
        Node::new(Allele::Input { label: 0 }, 0_usize, 0.3_f32, 1_i32, 99);
    }

    #[test]
    fn forward_jumper_connection() {
        Node::new(
            Allele::JumpForward { source_id: 0 },
            0_usize,
            0.3_f32,
            1_i32,
            1,
        );
    }

    #[test]
    fn recurrent_jumper_connection() {
        Node::new(
            Allele::JumpRecurrent { source_id: 0 },
            0_usize,
            0.3_f32,
            1_i32,
            1,
        );
    }

    #[test]
    fn not_a_node() {
        Node::new_nan(0 as usize, 1);
    }
}
