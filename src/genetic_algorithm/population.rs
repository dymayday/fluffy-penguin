//! Population doc string.

use genetic_algorithm::individual::Specimen;

#[derive(Debug)]
pub struct Population<T> {
    pub species: Vec<Specimen<T>>,
    pub current_generation: usize,
    // The structural mutation probability 'pm' , which is usually set between 5 and 10%.
    pub pm: T,
    // Global Innovation Number.
    gin: usize,
}

impl Population<f32> {
    pub fn new(
        population_size: usize,
        input_size: usize,
        output_size: usize,
        mutation_probability: f32,
    ) -> Self {
        let mut species: Vec<Specimen<f32>> = Vec::with_capacity(population_size);

        for _ in 0..population_size {
            species.push(Specimen::new(input_size, output_size));
        }
        let gin: usize = species[0].ann.genome.last().unwrap().gin;
        println!("Init GIN = {}", gin);

        Population {
            species,
            current_generation: 0,
            pm: mutation_probability,
            gin,
            // gin: input_size * output_size,
        }
    }


    /// New population of only specimen from the example.
    pub fn new_from_example(
        population_size: usize,
        mutation_probability: f32,
    ) -> Self {
        let mut species: Vec<Specimen<f32>> = Vec::with_capacity(population_size);

        for _ in 0..population_size {
            species.push(Specimen::new_from_example());
        }

        Population {
            species,
            current_generation: 0,
            pm: mutation_probability,
            gin: 11,
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
        let mut gin = self.gin;
        for mut specimen in &mut self.species {
            gin = specimen.structural_mutation(self.pm, gin);
            // println!("\n>> GIN = {}\n", gin);
        }
        self.gin = gin;
    }


    /// Apply evolution to our population by selection and reproduction.
    pub fn evolve(&mut self) {
        // self.selection();
        self.crossover();
        // unimplemented!();
    }

    /// Selection
    fn selection(&mut self) {
        unimplemented!();
    }


    /// Crossover is the main method of reproduction of our genetic algorithm.
    fn crossover(&mut self) {

        let mut offspring_vector: Vec<Specimen<f32>> = Vec::with_capacity(self.species.len());
        offspring_vector.push(self.species[0].clone());

        for specimen in &self.species[1..] {
            let mut offspring: Specimen<f32> = Specimen::crossover(&self.species[0], &specimen);
            offspring.update();

            offspring_vector.push(offspring);

        }

        self.species = offspring_vector;

    }


    /// Visualisation of the artificial neural network of each specimen of our population with
    /// GraphViz.
    pub fn render(&self, root_path: &str, print_weights: bool) {
        use rayon::prelude::*;
        use std::path::Path;

        // Specimen rendering are done in parallele.
        self.species
            .par_iter()
            .enumerate()
            .for_each(|(i, specimen)| {
                let file_name: String = format!("Specimen_{:03}.dot", i);
                let file_path = Path::new(root_path).join(&file_name);
                let file_path: &str = file_path
                    .to_str()
                    .expect("Fail to build render's output file path.");

                specimen.render(file_path, &format!("Specimen_{:03}", i), print_weights);
            });
    }
}
