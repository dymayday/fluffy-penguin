//! Here we declare all the activation function we might use in our artificial neural network,
//! namely: ISRLU, ISRU, ReLu, and Sigmoids.
//! See https://en.wikipedia.org/wiki/Activation_function for more.

pub trait TransferFunctionTrait<T: Copy> {
    fn tan_h(&self) -> T;

    fn soft_sign(&self) -> T;
}

impl TransferFunctionTrait<f32> for f32 {
    fn tan_h(&self) -> f32 {
        self.tanh()
    }

    fn soft_sign(&self) -> f32 {
        self / (1.0 + self.abs())
    }
}
