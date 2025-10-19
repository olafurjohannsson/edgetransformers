//! Feed-forward network implementation

use anyhow::Result;
use ndarray::{Array1, Array2, Array3};
use crate::utils::linear_algebra::matmul_3d_2d;
use crate::activations::gelu;

/// Feed-forward network with two linear layers
pub struct FeedForward {
    pub dense1_weight_t: Array2<f32>,
    pub dense1_bias: Array1<f32>,
    pub dense2_weight_t: Array2<f32>,
    pub dense2_bias: Array1<f32>,
}

impl FeedForward {
    pub fn new(
        dense1_weight: Array2<f32>,
        dense1_bias: Array1<f32>,
        dense2_weight: Array2<f32>,
        dense2_bias: Array1<f32>,
    ) -> Self {
        Self {
            dense1_weight_t: dense1_weight.t().to_owned(),
            dense1_bias,
            dense2_weight_t: dense2_weight.t().to_owned(),
            dense2_bias,
        }
    }
    
    pub fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        // First linear layer
        let mut intermediate = matmul_3d_2d(hidden, &self.dense1_weight_t);
        intermediate += &self.dense1_bias;
        
        // Activation
        gelu(&mut intermediate);
        
        // Second linear layer
        let mut output = matmul_3d_2d(&intermediate, &self.dense2_weight_t);
        output += &self.dense2_bias;
        
        Ok(output)
    }
}