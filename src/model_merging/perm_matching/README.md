# Permutation Matching Module

This module provides functionality for aligning model parameters through permutation matching before computing mergeability metrics.

## Overview

When fine-tuning neural networks from a common initialization, the resulting models may have neurons that correspond to each other but are in different orders (permuted). This permutation invariance can make it difficult to compare models directly. Permutation matching finds an optimal alignment between models to enable more accurate comparisons.

## Supported Architectures

Permutation matching is now supported for the following architectures:
- **CLIP ViT (OpenCLIP)**: ViT-B-32, ViT-B-16, ViT-L-14, etc. (OpenCLIP visual encoders)
- **Standard ViT**: timm/torchvision-style Vision Transformers
- **ResNet20**: Residual networks with 20 layers
- **ResNet50**: Residual networks with 50 layers
- **VGG16**: VGG with 16 layers

The implementation now includes a dedicated permutation spec for OpenCLIP ViT models, which handle the specific parameter naming conventions of CLIP's visual encoder (e.g., `transformer.resblocks.X.attn.*`).

For unsupported architectures or models with non-matching parameter names, the feature will gracefully fall back to using unaligned task vectors.

## Implementation

The implementation is based on the Git Re-Basin approach, which iteratively solves linear assignment problems to find permutations that maximize similarity between corresponding layers.

### Key Components

1. **`weight_matching.py`**: Core algorithm for finding optimal permutations
   - `weight_matching_for_task_vectors()`: Main function that performs iterative matching on task vectors

2. **`matcher.py`**: High-level interface
   - `apply_permutation_to_task_vectors()`: Applies permutation matching to align multiple task vectors

3. **`utils.py`**: Utility functions
   - Permutation matrix conversion functions
   - Parameter permutation application
   - Similarity computation

## Usage

### In `compute_mergeability.py`

Enable permutation matching by setting the flag in the configuration:

```bash
python scripts/compute_mergeability.py mergeability.perm_matching=true
```

Or modify the configuration file (`conf/multitask.yaml`):

```yaml
mergeability:
  perm_matching: true
  perm_matching_max_iter: 100  # Optional: adjust max iterations
```

### Configuration Options

- `perm_matching` (bool): Enable/disable permutation matching (default: `false`)
- `perm_matching_max_iter` (int): Maximum iterations for the matching algorithm (default: `100`)

## How It Works

1. **Load Models**: Fine-tuned models and pretrained model are loaded
2. **Compute Task Vectors**: Differences between fine-tuned and pretrained models
3. **Select Reference**: First task vector is used as the reference (fixed)
4. **Align Others**: Each other task vector is permuted to align with the reference
5. **Iterative Optimization**:
   - For each permutation layer, compute similarity matrix
   - Solve linear assignment problem to find best permutation
   - Apply permutation and repeat until convergence
6. **Compute Metrics**: Mergeability metrics are computed on aligned task vectors

## Benefits

- **Better Comparisons**: Aligned models enable more meaningful metric computations
- **Improved Predictions**: Metrics computed after alignment may better predict merge performance
- **Automatic**: No manual intervention required once enabled

## Reference

This implementation is adapted from the cycle-consistent model merging repository and based on:

- Git Re-Basin: Merging Models modulo Permutation Symmetries
- Cycle-Consistent Model Merging techniques

## Notes

- Permutation matching adds computational overhead (typically 1-2 minutes per pair)
- The first dataset in the list is used as the reference; all others are aligned to it
- Results include a `perm_matching` field indicating whether alignment was used
- If the model architecture is not supported, the feature automatically falls back to computing metrics on unaligned task vectors
- For ViT models, the implementation automatically detects the depth (12 for base/small, 24 for large) based on the model name

## Limitations

- **Permutation specifications must be manually defined** for each architecture and naming convention
- **CLIP ViT spec is experimental**: The CLIP ViT permutation spec has been created based on architectural knowledge but may need refinement based on actual results
- The CLIP spec currently handles the visual encoder portion only (not the text encoder)
- Creating a permutation spec requires deep understanding of the model architecture and how layers connect
- Some complex architectures may not have permutation specs defined yet
- Permutation matching is computationally expensive (adds 1-2 minutes per model pair)

## Creating Custom Permutation Specs

The CLIP ViT spec implementation (`clip_vit_spec.py`) serves as a reference for creating specs for other architectures:

1. **Inspect parameter names**: Print the keys of your model's state dict
2. **Understand layer structure**: Identify which layers can be permuted (typically hidden dimensions between layers)
3. **Define the spec**: Create a new `PermutationSpecBuilder` subclass
4. **Map parameters to permutations**: For each parameter, specify which permutation matrices affect which axes
5. **Add to matcher**: Update `_get_permutation_spec_for_model()` in `matcher.py`

The CLIP ViT spec handles these parameter naming patterns:
- `conv1.weight`: Initial patch embedding
- `ln_pre.weight/bias`: Pre-transformer layer norm
- `transformer.resblocks.{i}.ln_1.weight/bias`: Block layer norms
- `transformer.resblocks.{i}.attn.in_proj_weight/bias`: Attention projections
- `transformer.resblocks.{i}.mlp.c_fc.weight/bias`: MLP layers
- `ln_post.weight/bias`: Post-transformer layer norm

See `clip_vit_spec.py` for the complete implementation.
