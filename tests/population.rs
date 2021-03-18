//! Population tests

extern crate fluffy_penguin;
extern crate rand;

#[cfg(test)]
mod population {
    use fluffy_penguin::genetic_algorithm::Population;

    #[test]
    fn init_population() {
        let population_size = 10;
        let input_size = 2;
        let output_size = 1;

        // Create a new population with 10 individuals with 2 inputs, 1 output and a mutation
        // probability of 10%.
        let mut population = Population::new(population_size, 2, 1, 0.10, true);
        // Next we set the 'lambda' attribute to a proper value, usually between half the
        // population size and half of it.
        population.set_lambda(10);
        // And finally we set the 's rank' attribute.
        population.set_s_rank(1.5);

        let species_size = population.species.len();
        assert_eq!(species_size, population_size);

        for specimen in population.species.iter() {
            assert_eq!(input_size, specimen.input_size);
            assert_eq!(output_size, specimen.output_size);
        }
    }

    #[test]
    fn save_load() {
        use rand::{thread_rng, Rng};
        use std::fs;

        let file_name = "/tmp/pop_test.bc";
        let population_size: usize = 10;
        let mut population = Population::new(population_size, 2, 1, 0.10, true);

        // Build some random inputs to feed the model with.
        let inputs: Vec<[f32; 2]> = (0..population_size)
            .map(|_| {
                [
                    thread_rng().gen_range(-10_f32, 10_f32),
                    thread_rng().gen_range(-10_f32, 10_f32),
                ]
            })
            .collect();

        // Compute the model's outputs.
        let outputs: Vec<Vec<f32>> = population
            .species
            .iter_mut()
            .zip(&inputs)
            .map(|(specimen, input)| {
                specimen.update_input(input);
                specimen.evaluate()
            })
            .collect();

        // Save and load the population.
        population
            .save_to_file(file_name)
            .expect("Fail to save population to file.");
        let mut loaded_population =
            Population::load_from_file(file_name).expect("Fail to load Population from file.");

        // Checks if the loaded population compute the same results from before.
        let new_outputs: Vec<Vec<f32>> = loaded_population
            .species
            .iter_mut()
            .zip(&inputs)
            .map(|(specimen, input)| {
                specimen.update_input(input);
                specimen.evaluate()
            })
            .collect();

        // Clean up the mess we just did before exiting.
        fs::remove_file(file_name).expect("Failed to remove temporary test file.");

        assert_eq!(
            outputs, new_outputs,
            "Computed outputs before and after save/load are not equal."
        )
    }
}
