/*
Train neural network on the mountain car environment
with discrete action space
*/
extern crate gym_rs;
extern crate fluffy_penguin;
extern crate rayon;

use gym_rs::{GymEnv, PendulumEnv, Viewer, scale};
use self::gym_rs::ActionType;
use fluffy_penguin::genetic_algorithm::Population;
use fluffy_penguin::genetic_algorithm::individual::Specimen;
use rayon::prelude::*;
use std::cmp::Ordering;

struct MyEnv{}

impl MyEnv {
    fn get_fitness(&self, specimen: &mut Specimen<f32>) -> f32 {
        let mut env = PendulumEnv::default();
        let mut state: Vec<f64> = env.reset();

        let mut end: bool = false;
        let mut total_reward: f32 = 0.0;
        let mut episode_length :usize = 0;
        while !end {
            if episode_length > 200 {
                break
            }
            // normalize state
            let inputs: [f32; 3] = [
                state[0] as f32,
                state[1] as f32,
                state[2] as f32,
            ];
            specimen.update_input(&inputs);
            let output = specimen.evaluate();

            let action: ActionType = ActionType::Continuous(vec![output[0] as f64 * 2.0]);

            let (s, reward, done, _) = env.step(action);
            end = done;
            total_reward += reward as f32;
            state = s;

            episode_length += 1;
        }

        total_reward
    }

    fn evaluate(&self, specimen: &mut Specimen<f32>) -> f32 {
        // Get the avg score of 10 sample runs
        (0..10).collect::<Vec<usize>>().iter().map(|_| self.get_fitness(specimen)).sum::<f32>() / 10.0
    }
}

fn render_champion(champion: &mut Specimen<f32>) {
    let mut env = PendulumEnv::default();
    let mut state: Vec<f64> = env.reset();

    let mut viewer = Viewer::default();

    let mut end: bool = false;
    let mut episode_length: usize = 0;
    while !end {
        if episode_length > 200 {
            break;
        }
        // normalize state
        let inputs: [f32; 3] = [
            state[0] as f32,
            state[1] as f32,
            state[2] as f32,
        ];
        champion.update_input(&inputs);
        let output = champion.evaluate();

        let action: ActionType = ActionType::Continuous(vec![output[0] as f64 * 2.0]);

        let (s, _reward, done, _) = env.step(action);
        end = done;
        state = s;

        episode_length += 1;

        env.render(&mut viewer);
    }
}

fn main() {
    let pop_size: usize = 100;
    let mut pop = Population::new(pop_size, 3, 1, 0.15);
    pop.set_s_rank(2.0)
        .set_lambda(pop_size / 2);

    let env = MyEnv{};
    let cycle_stop: usize = 100;
    let cycle_per_structure = cycle_stop / 10;
    let mut champion: Specimen<f32> = pop.species[0].clone();
    champion.fitness = std::f32::MIN;
    for i in 0..cycle_stop {
        if i % cycle_per_structure == 0 {
            pop.exploration();
        } else {
            pop.exploitation();
        }

        let _scores: Vec<f32> = pop.species
            .par_iter_mut()
            .map(|s| {
                let fit = env.evaluate(s);
                s.fitness = fit;
                fit
            })
            .collect();

        pop.species.iter().for_each(|s| {
            if s.fitness > champion.fitness {
                champion = s.clone();
            }
        });

        if i + 1 < cycle_stop {
            pop.evolve();
        }

        println!("gen: {} || champion fitness: {}", i, champion.fitness);
    }
    println!("champion: {:?}", champion);

    render_champion(&mut champion);
}