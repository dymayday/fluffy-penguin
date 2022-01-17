//! This example demonstrate how well the algorithm performs on the traditional deep learning MNIST
//! problem.
//! Some work are still needed to tests the model on the a test portion of the dataset.

extern crate fluffy_penguin;
extern crate rayon;
extern crate rulinalg;
extern crate vision;

use fluffy_penguin::genetic_algorithm::{individual::Specimen, Population};
use rayon::prelude::*;
use rulinalg::utils;
use std::thread;
use vision::mnist::MNISTBuilder;

const STACK_SIZE: usize = 1000 * 2048 * 2048;
const DATASET_ROOT_PATH: &str = "tmp/mnist/";

const ROWS: usize = 28;
const COLS: usize = 28;
const LABEL_SIZE: usize = 10;
const DATASET_SIZE: usize = 1_000;
const TEST_DATASET_SIZE: usize = 1_000;


struct MnistDataset {
    train_imgs: [[f32; ROWS * COLS]; DATASET_SIZE],
    train_labels: [[f32; LABEL_SIZE]; DATASET_SIZE],
    #[allow(dead_code)]
    test_imgs: [[f32; ROWS * COLS]; TEST_DATASET_SIZE],
    #[allow(dead_code)]
    test_labels: [[f32; LABEL_SIZE]; TEST_DATASET_SIZE],
}


/// Loads the dataset into Vector of f32 values.
fn load_dataset() -> MnistDataset {
    let builder = MNISTBuilder::new();
    let mnist = builder
        .data_home(DATASET_ROOT_PATH)
        .verbose()
        .get_data()
        .unwrap_or_else(|_| panic!("Fail to build MNIST Dataset."));

    println!("train_imgs len = {}", mnist.train_imgs.len());
    println!("test_imgs len = {}", mnist.test_imgs.len());
    println!("train_labels len = {}", mnist.train_labels.len());
    println!("test_labels len = {}", mnist.test_labels.len());


    // Let's put everything on the stack for better performance.
    let mut train_imgs: [[f32; ROWS * COLS]; DATASET_SIZE] = [[0_f32; ROWS * COLS]; DATASET_SIZE];
    let mut train_labels: [[f32; LABEL_SIZE]; DATASET_SIZE] = [[0_f32; LABEL_SIZE]; DATASET_SIZE];

    // Cast u8 to f32 and build dataset output vectors.
    for image_idx in 0..DATASET_SIZE {
        for i in 0..mnist.train_imgs[image_idx].len() {
            train_imgs[image_idx][i] = f32::from(mnist.train_imgs[image_idx][i])
        }
        train_labels[image_idx][mnist.train_labels[image_idx] as usize] = 1_f32;
    }


    // Let's put everything on the stack for better performance.
    let mut test_imgs: [[f32; ROWS * COLS]; TEST_DATASET_SIZE] =
        [[0_f32; ROWS * COLS]; TEST_DATASET_SIZE];
    let mut test_labels: [[f32; LABEL_SIZE]; TEST_DATASET_SIZE] =
        [[0_f32; LABEL_SIZE]; TEST_DATASET_SIZE];

    // Cast u8 to f32 and build dataset output vectors.
    for image_idx in 0..TEST_DATASET_SIZE {
        for i in 0..mnist.test_imgs[image_idx].len() {
            test_imgs[image_idx][i] = f32::from(mnist.test_imgs[image_idx][i])
        }
        test_labels[image_idx][mnist.test_labels[image_idx] as usize] = 1_f32;
    }

    MnistDataset {
        train_imgs,
        train_labels,
        test_imgs,
        test_labels,
    }
}


/// This score function is a first dumb implementation, and means pretty much nothing in its current state.
/// It needs a lot of work in my opinion.
fn compute_specimen_score(
    specimen: &Specimen<f32>,
    train_imgs: &[[f32; ROWS * COLS]; DATASET_SIZE],
    train_labels: &[[f32; LABEL_SIZE]; DATASET_SIZE],
) -> f32 {
    let mut specimen = specimen.clone();
    let mut fitness = 0_f32;

    // compute the squared difference between the spcimen ANN output and the model to fit
    // for each data point
    // let mut squared_errors: Vec<f32> = Vec::with_capacity(DATASET_SIZE * LABEL_SIZE);

    for (inputs, labels) in train_imgs.iter().zip(train_labels.iter()) {
        specimen.update_input(inputs);
        let specimen_output = specimen.evaluate();

        let model_argmax = utils::argmax(labels);
        let ann_argmax = utils::argmax(&specimen_output);

        // let resp: f32;
        if model_argmax.0 == ann_argmax.0 {
            // resp = 1.0;
            fitness += 1.0;
        }
        // } else {
        //     resp = 0.0;
        // }
        // squared_errors.push(resp);

        // // Gathering the right answer.
        // let model_output: Vec<f32> = labels.to_vec();
        //
        // // And compare it with the output computedd by our ANN.
        // for e in 0..LABEL_SIZE {
        //     // squared_errors.push((model_output[e] * 250.0 - specimen_output[e]).powf(2.0));
        //     let resp: f32;
        //     if specimen_output[e] > 1.0 {
        //         resp = 1.0;
        //     } else {
        //         resp = 0.0;
        //     }
        //     squared_errors.push( (model_output[e] - resp).powf(2.0) );
        // }
    }


    // // return the RMSE
    // let error_sum = squared_errors.iter().fold(0., |sum, err| sum + err);
    // error_sum / (dataset_size as f32)
    fitness
}


/// This is where the magic takes place.
fn train_model(md: &MnistDataset) {
    use std::cmp::Ordering;

    let train_imgs = md.train_imgs;
    let train_labels = md.train_labels;

    let population_size: usize = 32;
    let input_size: usize = ROWS * COLS;
    let output_size: usize = 10;
    let mutation_probability: f32 = 0.05;

    let mut generation_counter: usize = 0;
    let cycle_per_structure: usize = 250;
    let cycle_stop: usize = 1_000_000;

    let mut population: Population<f32> = Population::new(
        population_size,
        input_size,
        output_size,
        mutation_probability,
    );

    // 'S Rank' value should be between 1.5 and 2.0
    // Here we set it to 2.0, meaning a lower chance for the worst individuals in the population
    // to get a chance to participate in the crossover (reprodiction) process.
    population.set_s_rank(2.0);

    // We trigger a first structural mutation for a better population mixing.
    population.exploration();

    // Prints the header of our scoring implementation
    println!(
        "[{:>5}], \t{:>10}  , \t{:>10} , \t{:>10}",
        "counter", "best", "mean", "worst",
    );


    for _ in 0..cycle_stop {
        generation_counter += 1;

        let scores: Vec<f32> = population
            .species
            .par_iter()
            .map(|specimen| compute_specimen_score(specimen, &train_imgs, &train_labels))
            .collect();

        // Update fitness of each specimen.
        // High score needs to represent a better fitness.
        for i in 0..population_size {
            population.species[i].fitness = scores[i];
        }

        // Selection phase.
        population.evolve();

        // Lookup for some better weights.
        if generation_counter % cycle_per_structure == 0 {
            population.exploration();
        } else {
            population.exploitation();
        }


        let worst_score = scores
            .iter()
            .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Greater))
            .unwrap();
        let mean_score: f32 = scores.iter().sum::<f32>() / population_size as f32;
        let best_score = scores
            .iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Greater))
            .unwrap();

        println!(
            "[{:>5}], \t{:.6} , \t{:.6}, \t{:.6}",
            generation_counter, best_score, mean_score, worst_score,
        );

        if generation_counter % 500 == 0 {
            population
                .save_to_file(&format!(
                    "tmp/mnist/mnist_population_save_{:04}.bc",
                    generation_counter
                ))
                .unwrap_or_else(|_| panic!("Fail to save MNIST's Population."));
        }
        // if *best_score <= 0.0020 {
        // // if *best_score <= 500.0 {
        //     println!("Saving Population and exitting...");
        //     population.render("tmp/mnist/viz", false, false);
        //     population.save_to_file("tmp/mnist/mnist_population_save.bc")
        //         .unwrap_or_else(|_| panic!("Fail to save MNIST's Population."));
        //     break;
        // }
    }
}


fn print_mnist_img(j: usize, mnist: &MnistDataset) {
    // let's show the first values of the MNIST dataset to show how it's stored
    // in memory.
    let header: Vec<u8> = (0..10).map(|x| x as u8).collect();
    println!(">> label = {:?}", header);
    print!(
        ">> label = {:?}",
        mnist.train_labels[j]
            .iter()
            .map(|x| *x as u8)
            .collect::<Vec<u8>>()
    );
    for i in 0..ROWS * COLS {
        if i % COLS == 0 {
            println!();
        }
        print!("{:>4}", mnist.train_imgs[j][i]);
    }
    println!();
}


fn run() {
    let mnist = load_dataset();

    println!("Let's print the first 10 digits to check our dataset parsing.");
    for j in 0..10 {
        println!("[{:^3}]", j);
        print_mnist_img(j, &mnist);
        println!();
    }

    train_model(&mnist);
}


fn main() {
    // We want to allocate everything on the stack, so we need
    // to spawn a new thread with explicit stack size.
    let child = thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(run)
        .unwrap();

    // Wait for thread to join
    child.join().unwrap();
}
