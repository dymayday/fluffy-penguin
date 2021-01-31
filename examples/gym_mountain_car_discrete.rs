/*
Train neural network on the mountain car environment
with discrete action space
*/
extern crate fluffy_penguin;
extern crate gym_rs;
extern crate rayon;

use self::gym_rs::ActionType;
use fluffy_penguin::genetic_algorithm::individual::Specimen;
use fluffy_penguin::genetic_algorithm::Population;
use gym_rs::{scale, GymEnv, MountainCarEnv, Viewer};
use rayon::prelude::*;
use std::cmp::Ordering;

struct MyEnv {}

impl MyEnv {
    fn get_fitness(&self, specimen: &mut Specimen<f32>) -> f32 {
        let mut env = MountainCarEnv::default();
        let mut state: Vec<f64> = env.reset();

        let mut end: bool = false;
        let mut total_reward: f32 = 0.0;
        while !end {
            if env.episode_length() > 200 {
                break;
            }
            // normalize state
            let inputs: [f32; 2] = [
                scale::<f32>(-1.2, 1.2, -1.0, 1.0, state[0] as f32),
                scale::<f32>(-0.07, 0.07, -1.0, 1.0, state[1] as f32),
            ];
            specimen.update_input(&inputs);
            let output = specimen.evaluate();

            let action: ActionType = if output[0] < 0.33 {
                ActionType::Discrete(0)
            } else if output[0] > 0.33 {
                ActionType::Discrete(2)
            } else {
                ActionType::Discrete(1)
            };

            let (s, reward, done, _) = env.step(action);
            end = done;
            total_reward += reward as f32;
            state = s;
        }

        total_reward
    }

    fn evaluate(&self, specimen: &mut Specimen<f32>) -> f32 {
        // Get the avg score of 10 sample runs
        (0..10)
            .collect::<Vec<usize>>()
            .iter()
            .map(|_| self.get_fitness(specimen))
            .sum::<f32>()
            / 10.0
    }
}

fn render_champion(champion: &mut Specimen<f32>) {
    let mut env = MountainCarEnv::default();
    let mut state: Vec<f64> = env.reset();

    let mut viewer = Viewer::default();

    let mut end: bool = false;
    while !end {
        if env.episode_length() > 200 {
            break;
        }
        // normalize state
        let inputs: [f32; 2] = [
            scale::<f32>(-1.2, 1.2, -1.0, 1.0, state[0] as f32),
            scale::<f32>(-0.07, 0.07, -1.0, 1.0, state[1] as f32),
        ];
        champion.update_input(&inputs);
        let output = champion.evaluate();

        let action: ActionType = if output[0] < 0.33 {
            ActionType::Discrete(0)
        } else if output[0] > 0.33 {
            ActionType::Discrete(2)
        } else {
            ActionType::Discrete(1)
        };

        let (s, _reward, done, _) = env.step(action);
        end = done;
        state = s;

        env.render(&mut viewer);
    }
}

fn main() {
    let pop_size: usize = 100;
    let mut pop = Population::new(pop_size, 2, 1, 0.15);
    pop.set_s_rank(2.0).set_lambda(pop_size / 2);

    let env = MyEnv {};
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

        let _scores: Vec<f32> = pop
            .species
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
