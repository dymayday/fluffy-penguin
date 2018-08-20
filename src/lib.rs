//! This is the core of our A.I. engine. "EANT2 with prune" is the algorithm we choose to implement and some
//! documentation can be found in this repository.
//! We choose EANT2 because of its capability to evolve its own structure and prune the connections
//! that are unnecessary. I find it very cool and powerful and I invit you to learn more about this
//! beautiful peace of art =].

extern crate rand;
extern crate rayon;
pub mod utils;
pub mod activation;
pub mod cge;
pub mod genetic_algorithm;


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
            let (gin_tmp, nn_id_tmp) = specimen_mutated.structural_mutation(0.5, gin, nn_id).unwrap();
            gin = gin_tmp;
            nn_id = nn_id_tmp;
            specimen_mutated.update_input(&input_vector);

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
    fn evaluation() {
        assert_eq!(
            Network::build_from_example().evaluate().expect("Fail to compute output from Network's evaluation."),
            [0.65220004]);
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
        Node::new(Allele::JumpForward { source_id: 0 }, 0_usize, 0.3_f32, 1_i32, 1);
    }

    #[test]
    fn recurrent_jumper_connection() {
        Node::new(Allele::JumpRecurrent { source_id: 0 }, 0_usize, 0.3_f32, 1_i32, 1);
    }

    #[test]
    fn not_a_node() {
        Node::new_nan(0 as usize, 1);
    }
}
