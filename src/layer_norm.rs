//! Layer normalization implementation

use ndarray::{Array1, Array3, Axis};

/// Layer normalization
pub struct LayerNorm {
    pub weight: Array1<f32>,
    pub bias: Array1<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Array1<f32>, bias: Array1<f32>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }
    
    /// Apply layer norm to a 3D tensor
    pub fn forward_3d(&self, hidden: &Array3<f32>) -> Array3<f32> {
        let mean = hidden.mean_axis(Axis(2)).unwrap();
        let var = hidden.var_axis(Axis(2), 0.0);
        
        let mean_expanded = mean.insert_axis(Axis(2));
        let var_expanded = var.insert_axis(Axis(2));
        
        // Compute normalized values
        let inv_std = (&var_expanded + self.eps).mapv(|x| 1.0 / x.sqrt());
        (hidden - &mean_expanded) * &inv_std * &self.weight + &self.bias
    }
}