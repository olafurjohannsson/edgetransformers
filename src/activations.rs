//! Activation functions for transformers

use ndarray::{Array3, Array4, Axis};

#[cfg(not(target_arch = "wasm32"))]
use ndarray::parallel::prelude::*;

const SQRT_2_OVER_PI: f32 = 0.7978845608_f32;
const GELU_COEFF: f32 = 0.044715_f32;

/// Activation function types
pub enum Activation {
    Gelu,
    Relu,
    Tanh,
    Swish,
}

/// Apply GELU activation in-place
#[inline(always)]
pub fn gelu(x: &mut Array3<f32>) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        x.par_mapv_inplace(|val| {
            let val_squared = val * val;
            let val_cubed = val_squared * val;
            let inner = SQRT_2_OVER_PI * (val + GELU_COEFF * val_cubed);
            val * 0.5 * (1.0 + inner.tanh())
        });
    }
    
    #[cfg(target_arch = "wasm32")]
    {
        x.mapv_inplace(|val| {
            let val_squared = val * val;
            let val_cubed = val_squared * val;
            let inner = SQRT_2_OVER_PI * (val + GELU_COEFF * val_cubed);
            val * 0.5 * (1.0 + inner.tanh())
        });
    }
}

/// Compute softmax over the last dimension of a 4D tensor
#[inline(always)]
pub fn softmax(scores: &Array4<f32>) -> Array4<f32> {
    // Find max for numerical stability
    let max_vals = scores.fold_axis(Axis(3), f32::NEG_INFINITY, |&acc, &x| acc.max(x));
    let max_expanded = max_vals.insert_axis(Axis(3));
    
    // Compute exp(x - max)
    let mut result = scores - &max_expanded;
    result.mapv_inplace(f32::exp);
    
    // Normalize
    let sum_exp = result.sum_axis(Axis(3)).insert_axis(Axis(3));
    result /= &sum_exp;
    
    result
}

/// Apply ReLU activation
pub fn relu(x: &mut Array3<f32>) {
    x.mapv_inplace(|val| val.max(0.0));
}

/// Apply Swish/SiLU activation
pub fn swish(x: &mut Array3<f32>) {
    x.mapv_inplace(|val| val * (1.0 / (1.0 + (-val).exp())));
}