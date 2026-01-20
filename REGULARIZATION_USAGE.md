# Mergeability Regularization Usage Guide

## Overview

This implementation adds task-agnostic mergeability regularization to model finetuning. The regularization encourages properties that make models generally mergeable without requiring knowledge of the specific merge partner.

## Regularization Terms

### R2: Moderate Update Regularization
- **Formula**: `λ₂ * ||θ_finetuned - θ_pretrained||₂²`
- **Purpose**: Penalizes large deviations from pretrained weights
- **Motivation**: All merge methods show negative coefficients for `weight_l2_distance` in stability analysis
- **Effect**: Keeps model close to pretrained initialization, making it easier to merge with other models

### R3: Gradient Magnitude Regularization
- **Formula**: `λ₃ * ||∇θ L_task||₂²`
- **Purpose**: Encourages moderate gradient magnitudes
- **Motivation**: All merge methods show negative coefficients for `gradient_l2_distance` in stability analysis
- **Effect**: Prevents extreme gradient updates that would push model far from merge-compatible regions

## Configuration

Edit `/home/ubuntu/thesis/MM/model-merging/conf/multitask.yaml`:

```yaml
train_regularization:
  # R2: Moderate Update Regularization
  enable_moderate_update: false  # Set to true to enable
  lambda_moderate_update: 0.01   # Hyperparameter weight

  # R3: Gradient Magnitude Regularization
  enable_grad_magnitude: false   # Set to true to enable
  lambda_grad_magnitude: 0.001   # Hyperparameter weight
```

## Usage Examples

### Example 1: Baseline (No Regularization)
```yaml
train_regularization:
  enable_moderate_update: false
  lambda_moderate_update: 0.01
  enable_grad_magnitude: false
  lambda_grad_magnitude: 0.001
```

Run finetuning:
```bash
python scripts/finetune.py dataset=Cars
```

Model saved to: `/home/ubuntu/thesis/MM/model-merging/models/ViT-B-16/Cars/model.pt`

### Example 2: R2 Only (Moderate Update)
```yaml
train_regularization:
  enable_moderate_update: true
  lambda_moderate_update: 0.01
  enable_grad_magnitude: false
  lambda_grad_magnitude: 0.001
```

Run finetuning:
```bash
python scripts/finetune.py dataset=Cars
```

Model saved to: `/home/ubuntu/thesis/MM/model-merging/models/ViT-B-16/Cars_moderate_update/model.pt`

### Example 3: R3 Only (Gradient Magnitude)
```yaml
train_regularization:
  enable_moderate_update: false
  lambda_moderate_update: 0.01
  enable_grad_magnitude: true
  lambda_grad_magnitude: 0.001
```

Run finetuning:
```bash
python scripts/finetune.py dataset=Cars
```

Model saved to: `/home/ubuntu/thesis/MM/model-merging/models/ViT-B-16/Cars_grad_magnitude/model.pt`

### Example 4: Both R2 + R3
```yaml
train_regularization:
  enable_moderate_update: true
  lambda_moderate_update: 0.01
  enable_grad_magnitude: true
  lambda_grad_magnitude: 0.001
```

Run finetuning:
```bash
python scripts/finetune.py dataset=Cars
```

Model saved to: `/home/ubuntu/thesis/MM/model-merging/models/ViT-B-16/Cars_moderate_update_grad_magnitude/model.pt`

## Finetuning All 20 Tasks

To finetune all tasks with regularization:

1. Enable desired regularizations in `conf/multitask.yaml`
2. Run finetuning for each dataset in the N20 benchmark:

```bash
# List of N20 datasets
datasets=(
    "Cars" "CIFAR10" "CIFAR100" "DTD" "EMNIST"
    "EuroSAT" "FashionMNIST" "FER2013" "Flowers102" "Food101"
    "GTSRB" "KMNIST" "MNIST" "OxfordIIITPet" "PCAM"
    "RenderedSST2" "RESISC45" "SUN397" "SVHN" "STL10"
)

# Finetune each dataset
for dataset in "${datasets[@]}"; do
    echo "Finetuning $dataset..."
    python scripts/finetune.py dataset=$dataset
done
```

## Checkpoint Naming Convention

Checkpoints are saved with the following naming pattern:
```
/home/ubuntu/thesis/MM/model-merging/models/ViT-B-16/{dataset_name}{suffix}/model.pt
```

Where `{suffix}` is:
- `` (empty) - No regularization
- `_moderate_update` - R2 only
- `_grad_magnitude` - R3 only
- `_moderate_update_grad_magnitude` - Both R2 + R3

## Hyperparameter Tuning Recommendations

### Starting Values (Low Computational Overhead)
- `lambda_moderate_update: 0.01`
- `lambda_grad_magnitude: 0.001`

### Tuning Strategy
1. Start with R2 only, tune `lambda_moderate_update` in range [0.001, 0.1]
2. Add R3, tune `lambda_grad_magnitude` in range [0.0001, 0.01]
3. Monitor training metrics:
   - `loss/train/{task_name}` - Main task loss
   - `reg/moderate_update/{task_name}` - R2 regularization term
   - `reg/grad_magnitude/{task_name}` - R3 regularization term
   - `reg/total/{task_name}` - Total regularization loss
   - `acc/test/{task_name}` - Test accuracy (should not degrade significantly)

### Expected Behavior
- **Too high λ**: Task accuracy drops significantly (over-regularization)
- **Too low λ**: No effect on mergeability (under-regularization)
- **Optimal λ**: Slight task accuracy drop (<5%) with improved mergeability

## Validation Experiment

After finetuning models with regularization, validate improved mergeability:

1. Finetune 10 models: 5 with regularization, 5 without (baseline)
2. Merge each with 3-5 held-out partners using all 4 merge methods
3. Compare average merge performance (acc/test/avg)
4. Expected result: Regularized models show higher average mergeability

## Monitoring During Training

The following metrics are logged to W&B (or console if W&B disabled):

- `acc/train/{task_name}` - Training accuracy
- `acc/test/{task_name}` - Test accuracy
- `loss/train/{task_name}` - Training loss (includes regularization)
- `reg/moderate_update/{task_name}` - R2 loss value (if enabled)
- `reg/grad_magnitude/{task_name}` - R3 loss value (if enabled)
- `reg/total/{task_name}` - Total regularization loss

## Implementation Details

### Files Modified
1. `/home/ubuntu/thesis/MM/model-merging/conf/multitask.yaml` - Configuration
2. `/home/ubuntu/thesis/MM/model-merging/src/model_merging/model/image_classifier.py` - Regularization logic
3. `/home/ubuntu/thesis/MM/model-merging/scripts/finetune.py` - Config passing and checkpoint naming

### How It Works
1. Pretrained encoder weights are saved before training starts
2. During each training step:
   - Compute task loss (cross-entropy)
   - If R2 enabled: Add L2 distance between current and pretrained weights
   - If R3 enabled: Add L2 norm of gradients (if available)
   - Total loss = task_loss + λ₂*R2 + λ₃*R3
3. Optimizer updates parameters using total loss
4. Checkpoint saved with suffix indicating which regularizations were used

## Troubleshooting

### Issue: Regularization terms not appearing in logs
**Solution**: Check that `hasattr(cfg, 'train_regularization')` returns True. Ensure `train_regularization` block exists in `multitask.yaml`.

### Issue: Checkpoint saved without suffix despite regularization enabled
**Solution**: Verify both `enable_*` flags are set to `true` (not `True` or `1`).

### Issue: Training loss explodes
**Solution**: Regularization weights too high. Reduce `lambda_moderate_update` and `lambda_grad_magnitude` by factor of 10.

### Issue: No effect on mergeability
**Solution**: Regularization weights too low. Increase gradually and monitor task accuracy (should drop slightly but not catastrophically).