//! Here we declare all the activation function we might use in our artificial neural network,
//! namely: ISRLU, ISRU, ReLu, and Sigmoids.
//! See https://en.wikipedia.org/wiki/Activation_function for more.


pub trait TransferFunctionTrait<T: Copy> {
    const ALPHA: T;
    fn isrlu(&self, alpha: T) -> T;
    fn isru(&self, alpha: T) -> T;
    fn relu(&self) -> T;
    fn sigmoids(&self) -> T;
}

impl TransferFunctionTrait<f32> for f32 {
    /// With α = 1, ISRLU saturation approaches -1.
    /// With α = 3, the negative saturation is reduced, so a smaller portion of
    /// the back-propagated error signal will pass to the next layer.
    /// This allows the network to output sparse activations while preserving its ability to reactivate dead neurons.
    ///
    /// Future work will establish what deeper saturation (α < 1) is appropriate when applying
    /// ISRLU to self-normalizing neural networks (Klambauer et al., 2017 https://arxiv.org/abs/1706.02515).
    const ALPHA: f32 = 1.0;

    /// ISRLU or "inverse square root linear unit" as better performance than ELU
    /// but has many of the same benefits. ISRLU and ELU have similar curves
    /// and characteristics. Both have negative values, allowing them to push mean unit
    /// activation closer to zero, and bring the normal gradient closer to the unit
    /// natural gradient, ensuring a noise-robust deactivation state,
    /// lessening the over fitting risk.
    ///
    /// The ISRLU hyperparameter `α` (alpha) controls the value to which an ISRLU saturates
    /// for negative inputs.
    ///
    /// https://arxiv.org/pdf/1710.09967.pdf
    fn isrlu(&self, alpha: f32) -> f32 {
        if *self < 0.0 {
            *self / (1.0 + alpha * (*self * *self)).sqrt()
        } else {
            *self
        }
    }

    /// ISRU or "inverse square root unit" is a computationally efficient variant of ISRLU which can be
    /// used for RNNs. Many RNNs use either long short-term memory (LSTM) and gated recurrent units
    /// (GRU) which are implemented with tanh and sigmoid activation functions. ISRU has less com-
    /// putational complexity but still has a similar curve to tanh and sigmoid.
    ///
    /// The ISRU hyperparameter `α` (alpha) controls the value to which an ISRLU saturates for negative
    /// inputs.
    fn isru(&self, alpha: f32) -> f32 {
        *self / (1.0 + alpha * (*self * *self)).sqrt()
    }

    /// Advantage of ReLu:
    /// * No gradient vanishing problem, as Relu’s gradient is constant = 1
    /// * Sparsity. When W*x < 0, Relu gives 0, which means sparsity.
    /// * Less calculation load. This may be least important.
    fn relu(&self) -> f32 {
        if *self < 0.0 {
            0.0
        } else {
            *self
        }
    }

    /// Sigmoids activation function.
    /// it takes a real-valued number and “squashes” it into range between 0 and 1. In particular,
    /// large negative numbers become 0 and large positive numbers become 1. The sigmoid function has
    /// seen frequent use historically since it has a nice interpretation as the firing rate of a
    /// neuron: from not firing at all (0) to fully-saturated firing at an assumed maximum frequency (1).
    /// In practice, the sigmoid non-linearity has recently fallen out of favor and it is rarely ever
    /// used.
    fn sigmoids(&self) -> f32 {
        1.0 / (1.0 + (-*self).exp())
    }
}


/// Advantage of ReLu:
/// * No gradient vanishing problem, as Relu’s gradient is constant = 1
/// * Sparsity. When W*x < 0, Relu gives 0, which means sparsity.
/// * Less calculation load. This may be least important.
pub fn relu(x: f32) -> f32 {
    if x < 0_f32 {
        0_f32
    } else {
        x
    }
}


pub fn relu_f64(x: f64) -> f64 {
    if x < 0_f64 {
        0_f64
    } else {
        x
    }
}


/// A smooth approximation to ReLu.
pub fn softplus(x: f32) -> f32 {
    (1_f32 + (x).exp()).ln()
}


/// ISRLU or "inverse square root linear unit" as better performance than ELU
/// but has many of the same benefits. ISRLU and ELU have similar curves
/// and characteristics. Both have negative values, allowing them to push mean unit
/// activation closer to zero, and bring the normal gradient closer to the unit
/// natural gradient, ensuring a noise-robust deactivation state,
/// lessening the over fitting risk.
///
/// The ISRLU hyperparameter 'α' controls the value to which an ISRLU saturates
/// for negative inputs.
///
/// https://arxiv.org/pdf/1710.09967.pdf
pub fn isrlu(x: f32, alpha: f32) -> f32 {
    if x < 0_f32 {
        x / (1_f32 + alpha * (x * x)).sqrt()
    } else {
        x
    }
}


/// ISRU or "inverse square root unit" is a computationally efficient variant of ISRLU which can be
/// used for RNNs. Many RNNs use either long short-term memory (LSTM) and gated recurrent units
/// (GRU) which are implemented with tanh and sigmoid activation functions. ISRU has less com-
/// putational complexity but still has a similar curve to tanh and sigmoid.
///
/// The ISRU hyperparameter `α` (alpha) controls the value to which an ISRLU saturates for negative
/// inputs.
pub fn isru(x: f32, alpha: f32) -> f32 {
    x / (1_f32 + alpha * (x * x)).sqrt()
}


/// Sigmoids activation function.
/// it takes a real-valued number and “squashes” it into range between 0 and 1. In particular,
/// large negative numbers become 0 and large positive numbers become 1. The sigmoid function has
/// seen frequent use historically since it has a nice interpretation as the firing rate of a
/// neuron: from not firing at all (0) to fully-saturated firing at an assumed maximum frequency (1).
/// In practice, the sigmoid non-linearity has recently fallen out of favor and it is rarely ever
/// used.
pub fn sigmoids(x: f32) -> f32 {
    1_f32 / (1_f32 + (-x).exp())
}
