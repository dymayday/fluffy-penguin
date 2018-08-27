//! Population doc string.

use genetic_algorithm::individual::Specimen;
use rand::{thread_rng, Rng};

/// Rank base selection parameter.
/// The usual formula for calculating the selection probability for linear
/// ranking schemes is parameterised by a value s (1 < s ≤ 2).
const RANK_S: f32 = 2.0;
// const RANK_S: f32 = 1.5;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Population<T> {
    pub species: Vec<Specimen<T>>,
    pub current_generation: usize,
    // The structural mutation probability 'pm' , which is usually set between 5 and 10%.
    pub pm: T,
    // Global Innovation Number.
    gin: usize,
    // The unique ID of the next new Neuron added to the linear genome by structural mutation.
    nn_id: usize,
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

        Population {
            species,
            current_generation: 0,
            pm: mutation_probability,
            gin,
            nn_id: output_size,
        }
    }


    /// New population of only specimen from the example.
    pub fn new_from_example(population_size: usize, mutation_probability: f32) -> Self {
        let mut species: Vec<Specimen<f32>> = Vec::with_capacity(population_size);

        for _ in 0..population_size {
            species.push(Specimen::new_from_example());
        }

        Population {
            species,
            current_generation: 0,
            pm: mutation_probability,
            gin: 11,
            nn_id: 4,
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
        let mut nn_id = self.nn_id;

        for mut specimen in &mut self.species {
            let (gin_tmp, nn_id_tmp) = specimen
                .structural_mutation(self.pm, gin, nn_id)
                .unwrap_or((gin, nn_id));
            gin = gin_tmp;
            nn_id = nn_id_tmp;
        }
        self.gin = gin;
        self.nn_id = nn_id;
    }


    /// Apply evolution to our population by selection and reproduction.
    pub fn evolve(&mut self) {
        &self.clean_fitness();
        &self.sort_species_by_fitness();

        let mating_pool: Vec<Specimen<f32>> = Population::selection(&self.species);

        self.crossover(&mating_pool);
    }

    /// Selection process using Stochastic Universal Sampling by default.
    fn selection(species: &[Specimen<f32>]) -> Vec<Specimen<f32>> {
        Population::stochastic_universal_sampling_selection(species)
    }


    /// Stochastic Universal Sampling is a simple, single phase, O(N) sampling algorithm. It is
    /// zero biased, has Minimum Spread and will achieve all N sanples in a single traversal.
    /// However, the algorithm is strictly sequential.
    pub fn stochastic_universal_sampling_selection(
        species: &[Specimen<f32>],
    ) -> Vec<Specimen<f32>> {
        let lambda: usize = species.len();
        // let lambda: usize = species.len() / 2_usize;
        // let lambda: usize = species.len() * 2_usize;

        let ranking_vector: Vec<f32> = Population::ranking_selection(&species);

        let mut cumulative_probability_distribution: Vec<f32> =
            Vec::with_capacity(ranking_vector.len());
        let mut cumsum: f32 = 0.0;
        for rank in ranking_vector {
            cumsum += rank;
            cumulative_probability_distribution.push(cumsum);
        }

        let mut specimen_index: usize = 0;
        let mut i: usize = 0;
        let mut r: f32 = thread_rng().gen_range(0.0, 1.0 / lambda as f32);

        let mut mating_pool: Vec<Specimen<f32>> = Vec::with_capacity(species.len());

        while specimen_index < lambda {
            while r <= cumulative_probability_distribution[i] {
                mating_pool.push(species[i].clone());
                r += 1.0 / lambda as f32;
                specimen_index += 1;
            }

            i += 1;
        }

        mating_pool
    }


    /// For good number of selection method in genetic algorithm, the fitness needs to be > 0, so
    /// we up each individual fitness in the population by the absolut value of the lowest fitness.
    fn clean_fitness(&mut self) {
        use std::cmp::Ordering;

        let lowest_fitness: f32 = *self
            .species
            .iter()
            .map(|s| {
                if s.fitness.is_finite() {
                    s.fitness
                } else {
                    -999.0
                }
            }).collect::<Vec<f32>>()
            .iter()
            .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Greater))
            .unwrap_or(&0.0);

        if lowest_fitness < 0.0 {
            for mut specimen in &mut self.species {
                specimen.fitness += lowest_fitness.abs();
            }
        }
    }


    /// Sort Specimen by their fitness value.
    pub fn sort_species_by_fitness(&mut self) {
        &self.species.sort_by_key(|k| k.fitness as i32);
    }


    /// Ranking Selection.
    fn ranking_selection(species: &[Specimen<f32>]) -> Vec<f32> {
        // This parameter controls wether or not the worst Specimen should have a chance to be in
        // the mating pool. s = 2 means a really low chance.
        let s: f32 = RANK_S;
        let population_size: usize = species.len();
        let mu: f32 = population_size as f32;

        let mut ranking_vector: Vec<f32> = Vec::with_capacity(population_size);

        for i in 0..population_size {
            ranking_vector
                .push(((2.0 - s) / mu) + ((2.0 * i as f32 * (s - 1.0)) / (mu * (mu - 1.0))));
        }

        ranking_vector
    }


    /// Crossover is the main method of reproduction of our genetic algorithm.
    fn crossover(&mut self, mating_pool: &[Specimen<f32>]) {
        let offspring_size: usize = self.species.len();
        let mut offspring_vector: Vec<Specimen<f32>> = Vec::with_capacity(offspring_size);

        // here we need 2 pool of shuffle mating index to make them randomly have baby babies with
        // each other.
        let mating_pool_size: usize = mating_pool.len();

        while offspring_vector.len() < offspring_size {
            let mut shuffled_mating_pool_index_1: Vec<usize> = Vec::with_capacity(mating_pool_size);
            let mut shuffled_mating_pool_index_2: Vec<usize> = Vec::with_capacity(mating_pool_size);

            for i in 0..mating_pool_size {
                if mating_pool[i].fitness.is_finite() {
                    shuffled_mating_pool_index_1.push(i);
                    shuffled_mating_pool_index_2.push(i);
                }
            }

            thread_rng().shuffle(&mut shuffled_mating_pool_index_1);
            thread_rng().shuffle(&mut shuffled_mating_pool_index_2);

            for (i, j) in shuffled_mating_pool_index_1
                .iter()
                .zip(shuffled_mating_pool_index_2.iter())
            {
                if offspring_vector.len() == offspring_size {
                    break;
                }

                let father: &Specimen<f32>;
                let mother: &Specimen<f32>;

                if mating_pool[*i].fitness >= mating_pool[*j].fitness {
                    father = &mating_pool[*i];
                    mother = &mating_pool[*j];
                } else {
                    father = &mating_pool[*j];
                    mother = &mating_pool[*i];
                }


                let mut offspring: Specimen<f32> = Specimen::crossover(father, mother, false);
                if offspring.ann.is_valid() {
                    offspring_vector.push(offspring);
                } else {
                    use cge::Network;
                    println!("\n\n\n\nFather:");
                    Network::pretty_print(&father.ann.genome);
                    father.render("tmp/father.dot", "Father", false);
                    father.save_to_file("tmp/father.bc");

                    println!("Father's Father:");
                    Network::pretty_print(&father.parents[0].ann.genome);
                    father.parents[0].render("tmp/father-father.dot", "Father_Father", false);
                    father.parents[0].save_to_file("tmp/father-father.bc");
                    println!("Father's Mother:");
                    Network::pretty_print(&father.parents[1].ann.genome);
                    father.parents[1].render("tmp/father-mother.dot", "Father_Mother", false);
                    father.parents[1].save_to_file("tmp/father-mother.bc");

                    println!("\n\nMother:");
                    Network::pretty_print(&mother.ann.genome);
                    mother.render("tmp/mother.dot", "Mother", false);
                    mother.save_to_file("tmp/mother.bc");

                    println!("Mother's Father:");
                    Network::pretty_print(&mother.parents[0].ann.genome);
                    mother.parents[0].render("tmp/mother-father.dot", "Mother_Father", false);
                    mother.parents[0].save_to_file("tmp/mother-father.bc");
                    println!("Mother's Mother:");
                    Network::pretty_print(&mother.parents[1].ann.genome);
                    mother.parents[1].render("tmp/mother-mother.dot", "Mother_Mother", false);
                    mother.parents[1].save_to_file("tmp/mother-mother.bc");

                    println!("\n\nOffspring:");
                    Network::pretty_print(&offspring.ann.genome);
                    // offspring.ann.render_to_dot("tmp/offspring.dot", "Offspring", false);

                    offspring.ann.is_valid();
                    let mut offspring: Specimen<f32> = Specimen::crossover(father, mother, true);

                    panic!(
                        "father {} and mother {} failed to reproduce.",
                        father.fitness * 10_000.0,
                        mother.fitness * 10_000.0
                    );
                    println!(
                        "father {} and mother {} failed to reproduce.",
                        father.fitness, mother.fitness
                    );
                }
            }
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
