
use genetic_algorithm::individual::Specimen;

pub struct Population<T> {
    species: Vec<Specimen<T>>,
    current_generation: usize,
    max_generation: usize,
}

impl Population<f32> {
    fn new(population_size: usize, max_generation: usize, input_size: usize, output_size: usize) -> Self {
        let mut species: Vec<Specimen<f32>> = Vec::with_capacity(population_size);

        Population { 
            species,
            current_generation: 0,
            max_generation,
        }
    }


    /// The exploitation phase researches the optimal weight of each Node in the current artificial
    /// neural network.
    pub fn exploitation(&mut self) {
        // ...
    }
}
