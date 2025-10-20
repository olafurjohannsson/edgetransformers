//! Multi-head attention implementation

use crate::activations::softmax;
use crate::utils::linear_algebra::{apply_attention_mask, matmul_3d_2d, matmul_4d};
use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4, Axis, s};

/// Multi-head attention mechanism
pub struct MultiHeadAttention {
    pub query_weight_t: Array2<f32>,
    pub query_bias: Array1<f32>,
    pub key_weight_t: Array2<f32>,
    pub key_bias: Array1<f32>,
    pub value_weight_t: Array2<f32>,
    pub value_bias: Array1<f32>,
    pub output_weight_t: Array2<f32>,
    pub output_bias: Array1<f32>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale_factor: f32,
}

impl MultiHeadAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        query_weight: Array2<f32>,
        query_bias: Array1<f32>,
        key_weight: Array2<f32>,
        key_bias: Array1<f32>,
        value_weight: Array2<f32>,
        value_bias: Array1<f32>,
        output_weight: Array2<f32>,
        output_bias: Array1<f32>,
    ) -> Self {
        let head_dim = hidden_size / num_heads;
        let scale_factor = 1.0 / (head_dim as f32).sqrt();

        Self {
            query_weight_t: query_weight,
            query_bias,
            key_weight_t: key_weight,
            key_bias,
            value_weight_t: value_weight,
            value_bias,
            output_weight_t: output_weight,
            output_bias,
            num_heads,
            head_dim,
            scale_factor,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        // If this is Some, we perform cross-attention (encoder-decoder), else we perform self-attention (decoder)
        encoder_hidden_states: Option<&Array3<f32>>,
        attention_mask: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        let batch_size = hidden_states.shape()[0];
        let seq_len = hidden_states.shape()[1];

        // Linear projections with bias

        // Project Q from decoders hidden states
        let mut q = matmul_3d_2d(hidden_states, &self.query_weight_t);
        q += &self.query_bias;

        // Determine source
        let kv_source = encoder_hidden_states.unwrap_or(hidden_states);
        let mut k = matmul_3d_2d(kv_source, &self.key_weight_t);
        k += &self.key_bias;
        let mut v = matmul_3d_2d(kv_source, &self.value_weight_t);
        v += &self.value_bias;

        // Reshape for multi-head attention
        let q = q
            .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let k_seq_len = kv_source.shape()[1];
        let k = k
            .into_shape_with_order((batch_size, k_seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let v = v
            .into_shape_with_order((batch_size, k_seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        // Compute attention scores
        let mut scores = matmul_4d(&q, &k.permuted_axes([0, 1, 3, 2]));
        scores *= self.scale_factor;

        // Apply mask
        if let Some(mask) = attention_mask {
            scores = apply_attention_mask(scores, mask);
        }

        // Softmax
        let weights = softmax(&scores);

        // Apply attention to values
        let context = matmul_4d(&weights, &v);

        // Reshape back
        let context = context.permuted_axes([0, 2, 1, 3]);
        let context = context
            .as_standard_layout()
            .into_shape_with_order((batch_size, seq_len, self.num_heads * self.head_dim))?
            .to_owned();

        // Output projection
        let mut output = matmul_3d_2d(&context, &self.output_weight_t);
        output += &self.output_bias;

        Ok(output)
    }
}
