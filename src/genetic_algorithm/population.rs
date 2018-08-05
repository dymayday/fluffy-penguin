
use genetic_algorithm::individual::Specimen;

pub struct Population<T> {
    pub species: Vec<Specimen<T>>,
    pub current_generation: usize,
    // The structural mutation probability 'pm' , which is usually set between 5 and 10%.
    pub pm: T,
}

impl Population<f32> {
    fn new(population_size: usize, input_size: usize, output_size: usize, mutation_probability: f32) -> Self {
        let mut species: Vec<Specimen<f32>> = Vec::with_capacity(population_size);

        for _ in 0..population_size {
            species.push(Specimen::new(input_size, output_size));
        }

        Population { 
            species,
            current_generation: 0,
            pm: mutation_probability,
        }
    }


    /// The exploitation phase researches the optimal weight of each Node in the current artificial
    /// neural network.
    pub fn exploitation(&mut self) {
        // Parametric exploitation of ever specimen in our population.
        for mut specimen in &mut self.species {
            specimen.parametric_mutation();
        }
    }


    /// Exploration of structures is accomplished by structural mutation which is performed at
    /// larger timescale. It is used to create new species or introduce new structures. From each
    /// of the existing structures, a new structure is formed and added to the existing ones. The
    /// weights of the newly acquired structural parts of the new structure are initialized to zero
    /// so as not to form(get) a new structure whose fitness value is less than its parent.
    pub fn exploration(&mut self) {
        // ...
        for mut specimen in &mut self.species {
            specimen.structural_mutation(self.pm);
        }
    }


    /// Selection
    pub fn selection(&mut self) {
        // ...

    }
}
