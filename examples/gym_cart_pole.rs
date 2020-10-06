extern crate gym_rs;
extern crate fluffy_penguin;
extern crate rayon;

use gym_rs::{CartPoleEnv, GymEnv, ActionType, Viewer};
use fluffy_penguin::genetic_algorithm::Population;
use fluffy_penguin::genetic_algorithm::individual::Specimen;
use rayon::prelude::*;
use std::cmp::Ordering;

pub struct MyEnv{}

impl MyEnv {
    fn get_fitness(&self, specimen: &mut Specimen<f32>) -> f32 {
        let mut env = CartPoleEnv::default();
        let mut state: Vec<f64> = env.reset();

        let mut end: bool = false;
        let mut total_reward: f32 = 0.0;
        while !end {
            if total_reward > 200.0 {
                break
            }

            // TODO: normalize
            let inputs: [f32; 4] = [
                state[0] as f32,
                state[1] as f32,
                state[2] as f32,
                state[3] as f32,
            ];
            specimen.update_input(&inputs);
            let output = specimen.evaluate();

            let action: ActionType = if output[0] < -0.0 {
                ActionType::Discrete(0)
            } else {
                ActionType::Discrete(1)
            };
            let (s, reward, done, _info) = env.step(action);
            end = done;
            total_reward += reward as f32;
            state = s;
        }

        // println!("total_reward: {}", total_reward);
        total_reward
    }

    fn evaluate(&self, specimen: &mut Specimen<f32>) -> f32 {
        // Get the avg score of 10 sample runs
        (0..10).collect::<Vec<usize>>().iter().map(|_| self.get_fitness(specimen)).sum::<f32>() / 10.0
    }
}

fn render_champion(champion: &mut Specimen<f32>) {
    let mut env = CartPoleEnv::default();
    let mut state: Vec<f64> = env.reset();

    let mut viewer = Viewer::new(1080, 1080);

    let mut end: bool = false;
    let mut total_reward: f64 = 0.0;
    while !end {
        if total_reward > 300.0 {
            println!("win!!!");
            break;
        }

        let inputs: [f32; 4] = [
            state[0] as f32,
            state[1] as f32,
            state[2] as f32,
            state[3] as f32,
        ];
        champion.update_input(&inputs);
        let output = champion.evaluate();

        let action: ActionType = if output[0] < 0.0 {
            ActionType::Discrete(0)
        } else {
            ActionType::Discrete(1)
        };
        let (s, reward, done, _info) = env.step(action);
        end = done;
        total_reward += reward;
        state = s;

        env.render(&mut viewer);
    }
}

fn main() {
    let pop_size: usize = 100;
    let mut pop = Population::new(pop_size, 4, 1, 0.15);
    pop.set_s_rank(2.0)
        .set_lambda(pop_size / 2);

    let env = MyEnv{};
    let cycle_stop: usize = 25;
    let cycle_per_structure = cycle_stop / 10;
    let mut champion: Specimen<f32> = pop.species[0].clone();
    for i in 0..cycle_stop {
        if i % cycle_per_structure == 0 {
            pop.exploration();
        } else {
            pop.exploitation();
        }

        let scores: Vec<f32> = pop.species
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

        let best_score = scores.iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Greater))
            .unwrap();
        println!("gen: {} || best_score: {}", i, best_score);
    }
    println!("champion: {:?}", champion);

    render_champion(&mut champion);
}