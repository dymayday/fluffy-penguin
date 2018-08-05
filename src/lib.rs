//! This is the core of our A.I. engine. "EANT2 with prune" is the algorithm we choose to implement and some
//! documentation can be found in this repository.
//! We choose EANT2 because of its capability to evolve its own structure and prune the connections
//! that are unnecessary. I find it very cool and powerful and I invit you to learn more about this
//! beautiful peace of art =].

extern crate rand;
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
mod network {
    use cge::network::Network;

    #[test]
    fn evaluation() {
        assert_eq!(Network::build_from_example().evaluate(), [0.69_f32]);
    }
}


#[cfg(test)]
mod node_tests {
    use cge::node;

    #[test]
    fn neuron() {
        node::Node::new(node::Allele::Neuron, 0 as usize, 0.3_f32, 1_i32);
    }

    #[test]
    fn input() {
        node::Node::new(node::Allele::Input, 0 as usize, 0.3_f32, 1_i32);
    }

    #[test]
    fn forward_jumper_connection() {
        node::Node::new(node::Allele::JumpForward, 0 as usize, 0.3_f32, 1_i32);
    }

    #[test]
    fn recurrent_jumper_connection() {
        node::Node::new(node::Allele::JumpRecurrent, 0 as usize, 0.3_f32, 1_i32);
    }
}
