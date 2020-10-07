//! Population doc string.

use rand::{thread_rng, Rng, seq::SliceRandom};
use rayon::prelude::*;

use crate::genetic_algorithm::individual::Specimen;
use crate::error::*;


/// The number of concurrent process used during the visualisation export phase to SVG.
const EXPORT_CPU_COUNT: usize = 4;

/// Rank base selection parameter.
/// The usual formula for calculating the selection probability for linear
/// ranking schemes is parameterised by a value s (1 < s ≤ 2).
/// This parameter controls wether or not the worst Specimen should have a chance to be in
/// the mating pool. s = 2 means a really low chance.
const S_RANK: f32 = 2.0;
// const S_RANK: f32 = 1.5;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Population<T> {
    pub species: Vec<Specimen<T>>,
    pub generation_counter: usize,
    /// The structural mutation probability 'pm', which is usually set between 5 and 10%.
    pub pm: T,
    /// Global Innovation Number.
    gin: usize,
    /// The unique ID of the next new Neuron added to the linear genome by structural mutation.
    nn_id: usize,
    /// This determines how many individuals will take part in the mating pool.
    lambda: usize,
    /// This parameter controls wether or not the worst Specimen should have a chance to be in
    /// the mating pool. s = 2 means a really low chance.
    s_rank: f32,
}


impl Population<f32> {
    pub fn new(
        population_size: usize,
        input_size: usize,
        output_size: usize,
        mutation_probability: f32,
    ) -> Self {
        let species: Vec<Specimen<f32>> = (0..population_size)
            .map(|_| Specimen::new(input_size, output_size))
            .collect();
        let gin: usize = species[0].ann.genome.last().unwrap().gin;

        Population {
            species,
            generation_counter: 0,
            pm: mutation_probability,
            gin,
            nn_id: output_size,
            lambda: population_size,
            s_rank: S_RANK,
        }
    }


    /// New population of only specimen from the example.
    pub fn new_from_example(population_size: usize, mutation_probability: f32) -> Self {
        let species: Vec<Specimen<f32>> = (0..population_size)
            .map(|_| Specimen::new_from_example())
            .collect();

        Population {
            species,
            generation_counter: 0,
            pm: mutation_probability,
            gin: 11,
            nn_id: 4,
            lambda: population_size,
            s_rank: S_RANK,
        }
    }


    /// Update the default 's rank' base selection parameter.
    /// The usual formula for calculating the selection probability for linear
    /// ranking schemes is parameterised by a value s (1 < s ≤ 2).
    /// This parameter controls wether or not the worst Specimen should have a chance to be in
    /// the mating pool. s = 2 means a really low chance.
    pub fn with_s_rank(mut self, s_rank: f32) -> Self {
        self.s_rank = s_rank;
        self
    }


    /// Updates the parameter which determines how many individuals will take part in the mating pool.
    pub fn with_lambda(mut self, lamba: usize) -> Self {
        self.lambda = lamba;
        self
    }


    /// Shrinks the population size to fit a desired value.
    /// Warning: shrinking the population after an evolution process has occured will result in a
    /// duplication of some specimen if the new population size is smaller than the previous one.
    pub fn shrink_to(mut self, population_size: usize) -> Self {
        // If there is enought blood to feed our needs, we just cut through some species from the population.
        if population_size <= self.species.len() {
            self.species = self.species[..population_size].to_vec();
        } else {
            // But if there is not enough individual to satisfy our appetite, we need to cycle through
            // the ones we have in stock.

            let mut species_iter_cycle = self.species.into_iter().cycle();
            self.species = (0..population_size)
                .map(|_| {
                    species_iter_cycle
                        .next()
                        .expect("Fail to cycle through the vector of Panda.")
                        .to_owned()
                })
                .collect();
        }
        self
    }


    /// Modify the default Rank base selection parameter.
    /// The usual formula for calculating the selection probability for linear
    /// ranking schemes is parameterised by a value 1 < s ≤ 2.
    /// This parameter controls wether or not the worst Specimen should have a chance to be in
    /// the mating pool. s = 2 means a really low chance.
    pub fn set_s_rank(&mut self, s: f32) -> &mut Self {
        assert!(1.0 < s, "s value should be: 1 < s ≤ 2");
        assert!(s <= 2.0, "s value should be: 1 < s ≤ 2");
        self.s_rank = s;
        self
    }


    /// Sets the default lamba value.
    /// This determines how many individuals will take part in the mating pool
    /// and should be <= population size.
    pub fn set_lambda(&mut self, lamba: usize) -> &mut Self {
        assert!(
            lamba <= self.species.len(),
            "lamba should be <= population size."
        );
        self.lambda = lamba;
        self
    }


    /// The exploitation phase researches the optimal weight of each Node in the current artificial
    /// neural network.
    pub fn exploitation(&mut self) {
        // Parametric exploitation of ever specimen in our population.
        for specimen in &mut self.species {
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

        for specimen in &mut self.species {
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
        self.generation_counter += 1;
        &self.clean_fitness();
        &self.sort_species_by_fitness();

        let mating_pool: Vec<Specimen<f32>> =
            Population::selection(&self.species, self.lambda, self.s_rank);

        // self.crossover(&mating_pool);
        self.par_crossover(&mating_pool);
    }

    /// Selection process using Stochastic Universal Sampling by default.
    fn selection(species: &[Specimen<f32>], lambda: usize, s_rank: f32) -> Vec<Specimen<f32>> {
        Population::stochastic_universal_sampling_selection(species, lambda, s_rank)
    }


    /// Stochastic Universal Sampling is a simple, single phase, O(N) sampling algorithm. It is
    /// zero biased, has Minimum Spread and will achieve all N sanples in a single traversal.
    /// However, the algorithm is strictly sequential.
    pub fn stochastic_universal_sampling_selection(
        species: &[Specimen<f32>],
        lambda: usize,
        s_rank: f32,
    ) -> Vec<Specimen<f32>> {
        let ranking_vector: Vec<f32> = Population::ranking_selection(&species, s_rank);

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
            })
            .collect::<Vec<f32>>()
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
    fn ranking_selection(species: &[Specimen<f32>], s: f32) -> Vec<f32> {
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
    fn _crossover(&mut self, mating_pool: &[Specimen<f32>]) {
        let offspring_size: usize = self.species.len();
        let mut offspring_vector: Vec<Specimen<f32>> = Vec::with_capacity(offspring_size);

        // here we need 2 pool of shuffle mating index to make them randomly have babies with
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

            shuffled_mating_pool_index_1.shuffle(&mut thread_rng());
            shuffled_mating_pool_index_2.shuffle(&mut thread_rng());

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


                let mut offspring: Specimen<f32> = Specimen::crossover(father, mother);

                if offspring.ann.is_valid() {
                    offspring_vector.push(offspring);
                } else {
                    use std::io::{stderr, Write};
                    writeln!(
                        stderr(),
                        "father {} and mother {} failed to reproduce.",
                        father.fitness,
                        mother.fitness
                    ).expect("Fail to write to 'stderr'");
                }
            }
        }

        self.species = offspring_vector;
    }


    /// Crossover is the main method of reproduction of our genetic algorithm.
    fn par_crossover(&mut self, mating_pool: &[Specimen<f32>]) {
        let offspring_size: usize = self.species.len();
        let mut offspring_vector: Vec<Specimen<f32>> = Vec::with_capacity(offspring_size);

        // here we need 2 pool of shuffle mating index to make them randomly have babies with
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

            shuffled_mating_pool_index_1.shuffle(&mut thread_rng());
            shuffled_mating_pool_index_2.shuffle(&mut thread_rng());

            let mut offspring_vector_tmp: Vec<Specimen<f32>> = shuffled_mating_pool_index_1
                .par_iter()
                .zip(shuffled_mating_pool_index_2.par_iter())
                .map(|(i, j)| {
                    if offspring_vector.len() == offspring_size {
                        return None;
                    } else {
                        let father: &Specimen<f32>;
                        let mother: &Specimen<f32>;

                        if mating_pool[*i].fitness >= mating_pool[*j].fitness {
                            father = &mating_pool[*i];
                            mother = &mating_pool[*j];
                        } else {
                            father = &mating_pool[*j];
                            mother = &mating_pool[*i];
                        }

                        let mut offspring: Specimen<f32> = Specimen::crossover(father, mother);

                        if offspring.ann.is_valid() {
                            return Some(offspring);
                        } else {
                            use std::io::{stderr, Write};
                            writeln!(
                                stderr(),
                                "father {} and mother {} failed to reproduce.",
                                father.fitness,
                                mother.fitness
                            ).expect("Fail to write to 'stderr'");
                            return None;
                        }
                    }
                })
                .collect::<Vec<Option<Specimen<f32>>>>()
                .iter()
                .filter_map(|s| s.to_owned())
                .collect();

            offspring_vector.append(&mut offspring_vector_tmp);
        }

        // If there is more offspring than we want, we simply skip the last ones.
        self.species = offspring_vector[..offspring_size].to_vec();
    }


    /// Visualisation of the artificial neural network of each specimen of our population with
    /// GraphViz.
    pub fn render(&self, root_path: &str, print_jumper: bool, print_weights: bool) {
        extern crate threadpool;

        use threadpool::ThreadPool;
        use std::path::Path;

        let pool = ThreadPool::new(EXPORT_CPU_COUNT);
        for (i, specimen) in self.species.clone().into_iter().enumerate() {
            let file_name: String = format!("Specimen_{:03}_g{:0>4}.dot", i, self.generation_counter);
            let file_path = Path::new(root_path).join(&file_name);
            let file_path: String = file_path.to_string_lossy().to_string();

            pool.execute(move || {
                match specimen.render(
                    &file_path,
                    &format!("Specimen_{:03}", i),
                    print_jumper,
                    print_weights,
                ) {
                    Some(_) => {},
                    None => panic!(format!("Fail to render Specimen {}.", i)),
                };
            });
        }

    }


    /// Save a Population to a file using 'Bincode' serialization
    /// https://github.com/TyOverby/bincode
    pub fn save_to_file(&self, file_name: &str) -> GenResult<()> {
        use bincode::serialize_into;
        use std::fs::File;
        use std::io::BufWriter;
        use crate::utils::create_parent_directory;


        create_parent_directory(file_name)?;

        let stream = BufWriter::new(File::create(file_name)?);
        serialize_into(stream, &self)?;

        Ok(())
    }


    /// Load a Specimen from a Bincode file.
    /// https://github.com/TyOverby/bincode
    pub fn load_from_file(file_name: &str) -> GenResult<Self> {
        use bincode::deserialize_from;
        use std::fs::File;
        use std::io::BufReader;

        let stream = BufReader::new(File::open(file_name)?);

        Ok(deserialize_from(stream)?)
    }
}
