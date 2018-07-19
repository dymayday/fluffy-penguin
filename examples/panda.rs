//! This example will actually be used during the dev process.

extern crate fluffy_penguin;

use fluffy_penguin::activation;

fn main() {
    
    for i in 0..21 {
        let i: f32 = i as f32 - 5_f32;
        // println!("{:4} : ReLu = {} - SoftPlus = {}", i, activation::relu(i), activation::softplus(i));
        
        println!("{:4} :", i);
        println!("     ReLu = {}", activation::relu_f32(i));
        println!("ISRLU 1.0 = {}", activation::isrlu(i, 1_f32));
        println!("ISRLU 3.0 = {}", activation::isrlu(i, 3_f32));
        println!(" SoftPlus = {}", activation::softplus(i));
        println!(" ISRU 1.0 = {}", activation::isru(i, 1_f32));
        println!(" ISRU 3.0 = {}", activation::isru(i, 3_f32));
        println!(" Sigmoids = {}", activation::sigmoids(i));
        println!("");
    }
}
