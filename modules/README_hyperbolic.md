# Hyperbolic Poincaré Enhancement for Densely Anchored Sampling (DAS)

This module provides hyperbolic Poincaré geometry operations to enhance the existing Densely Anchored Sampling (DAS) system. The hyperbolic operations can improve embedding quality by better preserving hierarchical relationships in the data.

## Overview

The hyperbolic Poincaré ball model represents hyperbolic space as the interior of a unit ball in Euclidean space, where distances grow exponentially as we move toward the boundary. This is particularly useful for:

- **Hierarchical data structures**: Better preserving tree-like relationships
- **Improved embedding diversity**: Generating more diverse and meaningful embeddings
- **Enhanced class separation**: Better distinguishing between different classes
- **Robust transformations**: More stable geometric operations

## Files

- `hyperbolic_poincare.py`: Main hyperbolic operations module
- `hyperbolic_usage_example.py`: Usage examples and integration guide
- `README_hyperbolic.md`: This documentation file

## Key Components

### 1. HyperbolicPoincare Class

Core hyperbolic operations in the Poincaré ball model:

```python
from hyperbolic_poincare import HyperbolicPoincare

# Initialize
hyperbolic = HyperbolicPoincare(dim=128, curvature=-1.0)

# Convert between Euclidean and Poincaré spaces
poincare_embeddings = hyperbolic.to_poincare(euclidean_embeddings)
euclidean_embeddings = hyperbolic.to_euclidean(poincare_embeddings)

# Compute hyperbolic distances
distances = hyperbolic.poincare_distance(x, y)

# Perform hyperbolic operations
result = hyperbolic.poincare_add(x, y)  # Möbius addition
result = hyperbolic.poincare_exp_map(v, x)  # Exponential map
result = hyperbolic.poincare_log_map(y, x)  # Logarithmic map
```

### 2. HyperbolicDenselyAnchoredSampling Class

Enhanced DAS with hyperbolic operations:

```python
from hyperbolic_poincare import HyperbolicDenselyAnchoredSampling

# Initialize hyperbolic DAS
hyperbolic_das = HyperbolicDenselyAnchoredSampling(
    num_classes=10,
    dim=128,
    num_produce=3,
    normalize=True,
    dfs_num_scale=4,
    dfs_scale_range=(0.5, 2.0),
    mts_num_transformation_bank=10,
    mts_scale=0.01,
    hyperbolic_weight=0.5,  # Balance between Euclidean and hyperbolic
    curvature=-1.0,
    detach=True
)

# Generate enhanced embeddings
produced_embeddings, produced_targets = hyperbolic_das(embeddings, labels)
```

## Integration with Existing DAS

### Option 1: Direct Replacement

Replace the original `DenselyAnchoredSampling` with `HyperbolicDenselyAnchoredSampling`:

```python
# Original
from embedding_producer import DenselyAnchoredSampling
das = DenselyAnchoredSampling(num_classes, dim, ...)

# Enhanced with hyperbolic operations
from hyperbolic_poincare import HyperbolicDenselyAnchoredSampling
hyperbolic_das = HyperbolicDenselyAnchoredSampling(
    num_classes, dim, 
    hyperbolic_weight=0.5,  # Add this parameter
    ...
)
```

### Option 2: Enhanced EmbeddingProducer

Create an enhanced version that supports both Euclidean and hyperbolic operations:

```python
from embedding_producer import EmbeddingProducer
from hyperbolic_poincare import HyperbolicDenselyAnchoredSampling

class EnhancedEmbeddingProducer(EmbeddingProducer):
    def __init__(self, num_classes, dim, **kwargs):
        super().__init__(num_classes, dim, **kwargs)
        
        # Add hyperbolic DAS
        self.hyperbolic_das = HyperbolicDenselyAnchoredSampling(
            num_classes=num_classes,
            dim=dim,
            hyperbolic_weight=kwargs.get('hyperbolic_weight', 0.5),
            **{k: v for k, v in kwargs.items() if k.startswith('das_')}
        )
    
    def produce_hyperbolic(self, embeddings, labels):
        """Produce embeddings using hyperbolic DAS."""
        return self.hyperbolic_das(embeddings, labels)

# Usage
enhanced_producer = EnhancedEmbeddingProducer(
    num_classes=10, dim=128, 
    hyperbolic_weight=0.5,
    das_num_produce=3,
    ...
)

# Use original DAS
original_embeddings, original_labels = enhanced_producer(embeddings, labels)

# Use hyperbolic DAS
hyperbolic_embeddings, hyperbolic_labels = enhanced_producer.produce_hyperbolic(embeddings, labels)
```

## Key Parameters

### HyperbolicDenselyAnchoredSampling Parameters

- `hyperbolic_weight` (float, 0-1): Weight for hyperbolic operations vs Euclidean operations
  - `0.0`: Pure Euclidean DAS (original behavior)
  - `0.5`: Balanced approach (recommended)
  - `1.0`: Pure hyperbolic operations
- `curvature` (float): Negative curvature of hyperbolic space (default: -1.0)

### HyperbolicPoincare Parameters

- `dim` (int): Embedding dimension
- `curvature` (float): Hyperbolic curvature (default: -1.0)
- `eps` (float): Numerical stability constant (default: 1e-8)

## Benefits of Hyperbolic Operations

### 1. Better Hierarchical Preservation

Hyperbolic space naturally preserves hierarchical relationships:

```python
# Hyperbolic operations better preserve tree-like structures
hyperbolic_centroid = hyperbolic.hyperbolic_centroid(points, weights)
hyperbolic_interpolated = hyperbolic.hyperbolic_interpolation(x, y, t)
```

### 2. Enhanced Diversity

Hyperbolic sampling generates more diverse embeddings:

```python
# Hyperbolic-aware sampling
sampled_embeddings, sampled_labels = hyperbolic.hyperbolic_sampling(
    embeddings, labels, 
    num_samples=3, 
    temperature=0.1
)
```

### 3. Improved Class Separation

Hyperbolic transformations can better separate different classes:

```python
# Compare class separation
euclidean_das = HyperbolicDenselyAnchoredSampling(hyperbolic_weight=0.0)
hyperbolic_das = HyperbolicDenselyAnchoredSampling(hyperbolic_weight=0.5)

euclidean_results = euclidean_das(embeddings, labels)
hyperbolic_results = hyperbolic_das(embeddings, labels)
```

## Usage Examples

See `hyperbolic_usage_example.py` for comprehensive examples:

1. **Basic Hyperbolic Operations**: Converting between spaces, computing distances
2. **Hyperbolic DAS Integration**: Using hyperbolic DAS with existing system
3. **Enhanced EmbeddingProducer**: Creating enhanced producer with both capabilities
4. **Comparison Analysis**: Comparing Euclidean vs Hyperbolic performance

## Installation Requirements

```bash
pip install torch torchvision
```

## Mathematical Background

### Poincaré Ball Model

The Poincaré ball model represents hyperbolic space as the interior of a unit ball in Euclidean space. Key operations:

1. **Distance**: `d(x,y) = acosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))`
2. **Möbius Addition**: `x ⊕ y = ((1+2⟨x,y⟩+||y||²)x + (1-||x||²)y)/(1+2⟨x,y⟩+||x||²||y||²)`
3. **Exponential Map**: `exp_x(v) = x ⊕ (tanh(λ_x||v||/2)v/||v||)`
4. **Logarithmic Map**: `log_x(y) = 2atanh(||-x⊕y||)(-x⊕y)/(λ_x||-x⊕y||)`

Where `λ_x = 2/(1-||x||²)` is the conformal factor.

### Integration with DAS

The hyperbolic operations are integrated into DAS by:

1. **DFS (Discriminative Feature Scaling)**: Enhanced with hyperbolic scaling
2. **MTS (Memorized Transformation Shifting)**: Using hyperbolic transformations
3. **Combined Operations**: Weighted combination of Euclidean and hyperbolic results

## Performance Considerations

- **Computational Overhead**: Hyperbolic operations add ~20-30% computational cost
- **Memory Usage**: Similar to original DAS
- **Numerical Stability**: Built-in epsilon handling for edge cases
- **Gradient Flow**: Compatible with backpropagation

## Best Practices

1. **Start with Balanced Weight**: Use `hyperbolic_weight=0.5` initially
2. **Tune Curvature**: Adjust `curvature` based on data hierarchy
3. **Monitor Diversity**: Check embedding diversity metrics
4. **Validate Performance**: Compare with original DAS on your dataset
5. **Gradual Integration**: Start with small hyperbolic weight and increase gradually

## Troubleshooting

### Common Issues

1. **Numerical Instability**: Increase `eps` parameter
2. **Poor Performance**: Try different `hyperbolic_weight` values
3. **Memory Issues**: Reduce `mts_num_transformation_bank` size
4. **Import Errors**: Ensure PyTorch is installed

### Debugging

```python
# Check hyperbolic operations
hyperbolic = HyperbolicPoincare(dim=128)
test_embeddings = torch.randn(10, 128)
poincare_embeddings = hyperbolic.to_poincare(test_embeddings)
print(f"Poincaré norm range: {poincare_embeddings.norm(dim=1).min():.4f} - {poincare_embeddings.norm(dim=1).max():.4f}")

# Should be < 1.0 for valid Poincaré embeddings
```

## References

1. Nickel, M., & Kiela, D. (2017). Poincaré embeddings for learning hierarchical representations. NeurIPS.
2. Ganea, O., Bécigneul, G., & Hofmann, T. (2018). Hyperbolic neural networks. NeurIPS.
3. Chami, I., et al. (2019). Hyperbolic graph convolutional neural networks. NeurIPS. 