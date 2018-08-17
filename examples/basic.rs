extern crate fluffy_penguin;
use fluffy_penguin::cge::Network;
use fluffy_penguin::genetic_algorithm::individual::Specimen;
use fluffy_penguin::genetic_algorithm::Population;

fn init_population(
    population_size: usize,
    input_size: usize,
    output_size: usize,
    mutation_probability: f32
    ) -> Population<f32> {
    let mut population: Population<f32> = Population::new(
        population_size,
        input_size,
        output_size,
        mutation_probability,
    );
    population
}

/// returns the indices of the `n` minimum values
fn n_argmin(data: &Vec<f32>, n: usize) -> Vec<usize> {
    assert!(n <= data.len());
    use std::f32;
    let mut indices: Vec<usize> = Vec::with_capacity(n);
    let mut last_min = f32::NEG_INFINITY;
    for i in 0..n {
        let (idx, value) = data.iter().enumerate().fold(
            (0, 100.),
            | (min_idx, min_value), (idx, value) | {
                if *value < min_value && *value > last_min { 
                    (idx, *value) 
                } else {
                    (min_idx, min_value)
                }
            }
        );
        last_min = value;
        indices.push(idx);
    }
    indices
}

/// non-linear problem
fn model_to_fit(a: f32, b: f32) -> f32 {
    assert!(a >= -1. && a <= 1.);
    assert!(b >= -1. && b <= 1.);
    (a.powf(2.) + b.powf(2.)).abs() / 2_f32
} 

/// test specimen on 100 data point, and return the mean squared 
/// error of the result
fn compute_specimen_score(specimen: &Specimen<f32>) -> f32 {
    let mut specimen = specimen.clone();
    let dataset_size: usize = 100;
    // Build a simple dataset for both inputs
    let dataset_a: Vec<f32> = (0..dataset_size).map(
            |i| (i as f32 + 0.5) * (2. / dataset_size as f32) - 1.
        ).collect();
    let dataset_b: Vec<f32> = (0..dataset_size).map(
            |i| 1. - (i as f32 + 0.5) * (2. / dataset_size as f32)
        ).collect();
    // compute the squared difference between the spcimen ANN output and the model to fit
    // for each data point
    let mut squared_errors: Vec<f32> = Vec::with_capacity(dataset_size);
    for i in 0..dataset_size {
        let (a, b) = (dataset_a[i], dataset_b[i]);
        specimen.update_input(&[a, b]);
        let specimen_output = specimen.evaluate()[0];
        let model_output = model_to_fit(a, b);
        squared_errors.push((model_output - specimen_output).powf(2.));
    }
    // return the RMSE
    let error_sum = squared_errors.iter().fold(0., |sum, err|  sum + err );
    error_sum / (dataset_size as f32)
}


///
/// Global algorythm:
///
/// // init step
/// build a set of individuals, with:
///     - the same ANN topography
///     - random weights
///
/// // evolution stepS
/// for i in 0..infinity {
///     
///     // selection step
///     apply fitness function on each individual
///     select `N` best, drop others
///
///     // reproduction step
///     for pair in pairs_in_set {
///         align each individual genome to each other
///         merge genomes => new individual
///     }
///     select new individuals and drop old ones
///
///     // mutation step
///     if i % (big_number) == 0 {
///         do structural mutation, ether:
///             - add a node
///             - add a connexion
///             - delete a connexion
///     } else {
///         do weight mutation
///     }
/// }
///
///
fn main() {
    let population_size = 10;
    // build population
    let mut population = init_population(
        population_size, // size of population
        2, // nb of input node in each ANN
        1, // nb of output node in each ANN
        0.5 // mutation probability
    );

    /* EVOLUTION */
    let mut generation_counter: i64 = 0;
    let cycle_per_structure = 500;
    loop {

        /* SELECTION */
        // Evalute each specimen against test data and then compute the fitness 
        // score of each specimen
        let scores: Vec<f32> = population.species.iter()
            .map(|specimen| compute_specimen_score(specimen))
            .collect();

        // select the best half of the population
        let mut bests: Vec<Specimen<f32>> = n_argmin(&scores, population_size /2)
            .iter()
            .map(|idx| population.species[*idx].clone())
            .collect();

        /* REPRODUCTION */
        // build new set
        let mut new_specimens: Vec<Specimen<f32>> = Vec::new();
        for i in 0..bests.len() {
            for j in (i+1)..bests.len() {
               new_specimens.push(
                   Specimen::crossover(
                       &bests[i],
                       &bests[j]
                   )
               ); 
            }
        }
        // drop old set of individuals, and replace it by their offspring
        population.species = new_specimens;

        /* MUTATION */
        // ann mutation
        if generation_counter % cycle_per_structure == 0 {
            for specimen in population.species.iter_mut() {
                specimen.parametric_mutation();
            }

        } else {
            population.exploration();
        };
                                         
        println!("{:?}", scores);
        generation_counter += 1;
    }
}

