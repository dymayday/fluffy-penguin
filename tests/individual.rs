//! Individuals tests

extern crate fluffy_penguin;

#[cfg(test)]
mod individual {
    use fluffy_penguin::cge::Network;
    use fluffy_penguin::genetic_algorithm::Specimen;

    #[test]
    fn crossover() {
        // Let's build the 2 Specimen from
        let spe1 = Specimen {
            input_size: 1_usize,
            output_size: 2_usize,
            ann: Network::<f32>::_build_parent1_from_example(),
            fitness: 1.0,
        };

        let spe2 = Specimen {
            input_size: 1_usize,
            output_size: 2_usize,
            ann: Network::<f32>::_build_parent2_from_example(),
            fitness: 0.0,
        };

        let _child = Specimen::crossover(&spe1, &spe2);
    }
}
