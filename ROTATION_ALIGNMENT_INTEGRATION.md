# Rotation Symmetry Alignment Integration

## Overview

Rotation symmetry alignment has been successfully integrated into the model-merging pipeline at line 120 of `evaluate_multitask_merging.py`.

## What Was Done

### 1. Backup Created
- **Commit**: `3f83f02` - "Backup before rotation alignment integration"
- **Revert command** (if needed): `git reset --hard 3f83f02^`

### 2. New Files Created

#### `/home/ubuntu/thesis/MM/model-merging/src/model_merging/alignment/__init__.py`
- Module initialization file
- Exports `apply_rotation_alignment` function

#### `/home/ubuntu/thesis/MM/model-merging/src/model_merging/alignment/rotation_alignment.py`
- Main alignment implementation (266 lines)
- Functions:
  - `apply_rotation_alignment()` - Main entry point for alignment
  - `align_clip_models()` - Core rotation alignment logic
  - `verify_alignment()` - Verification of weight changes and output invariance
  - `_state_dict_to_encoder()` - Helper to convert state_dicts to model objects

### 3. Modified Files

#### `/home/ubuntu/thesis/MM/model-merging/scripts/evaluate_multitask_merging.py`
- **Line 38**: Added import for `apply_rotation_alignment`
- **Lines 120-128**: Added alignment conditional block

```python
# Apply rotation symmetry alignment if enabled
if cfg.alignment:
    pylogger.info("Applying rotation symmetry alignment...")
    finetuned_models = apply_rotation_alignment(
        finetuned_state_dicts=finetuned_models,
        model_name=cfg.nn.encoder.model_name,
        device=cfg.device,
        logger=pylogger
    )
```

### 4. Configuration

#### `/home/ubuntu/thesis/MM/model-merging/conf/multitask.yaml`
- **Line 23**: `alignment: false` (already added by user)

## How It Works

1. **Model Loading** (lines 113-118): Models are loaded from HuggingFace as state_dicts
2. **Alignment** (lines 120-128, NEW): If `cfg.alignment=true`:
   - First model in the list becomes the anchor
   - All other models are aligned to the anchor using rotation symmetry
   - Verification checks that:
     - Weights changed (~36 parameters for attention layers)
     - Output invariance maintained (max diff < 1e-4)
3. **Merging** (line 137): Aligned models are merged normally

## Alignment Behavior

### Anchor Selection
- **First model** in `finetuned_models` dict is used as anchor
- Anchor model is **not modified**
- All other models are aligned to this anchor

### What Gets Aligned
- ✓ **Attention Q, K projections** (rotation via SVD)
- ✓ **Attention V, O projections** (rotation via SVD)
- ✓ **FFN layers** (permutation via Hungarian algorithm)
- ✗ **LayerNorm, embeddings** (not modified)

### Verification
For each aligned model, the system checks:
1. **Weight changes**: Should modify ~36 parameters (attention weights)
2. **Output invariance**: Max output difference should be < 1e-4

## Testing Commands

### Test 1: Default Behavior (Alignment Disabled)
```bash
cd /home/ubuntu/thesis/MM/model-merging
python scripts/evaluate_multitask_merging.py
```

**Expected**: Script runs normally, no alignment messages in logs.

### Test 2: With Alignment Enabled
```bash
cd /home/ubuntu/thesis/MM/model-merging
python scripts/evaluate_multitask_merging.py alignment=true
```

**Expected logs**:
```
======================================================================
ROTATION SYMMETRY ALIGNMENT
======================================================================
Selected anchor model: MNIST
Models to align: 1

Aligning CIFAR10 to MNIST...
  ✓ Weights changed (alignment applied)
  ✓ Output invariance verified: max diff = 1.96e-05

✓ Rotation alignment complete for 1 models!
======================================================================
```

### Test 3: With Specific Benchmark
```bash
cd /home/ubuntu/thesis/MM/model-merging
python scripts/evaluate_multitask_merging.py alignment=true benchmark=N2
```

### Test 4: Verify Alignment on Pairwise Merging
```bash
cd /home/ubuntu/thesis/MM/model-merging
python scripts/evaluate_multitask_merging.py alignment=true all_pairwise=true benchmark=N2
```

## Troubleshooting

### If alignment fails with import errors:
Check that ViT_ImageEncoder_RotationSymmetry exists at:
```bash
ls /home/ubuntu/thesis/MM/ViT_ImageEncoder_RotationSymmetry/src/alignment_utils.py
```

### If verification fails (no weights changed):
- Check that models are actually different (not same model loaded multiple times)
- Verify state_dicts have correct format with "model." prefix

### If output invariance fails (diff > 1e-4):
- This indicates a bug in the alignment implementation
- Check that rotation matrices are orthogonal
- Verify that attention/FFN updates are applied correctly

## Reverting Changes

If you need to revert to the state before integration:

```bash
cd /home/ubuntu/thesis/MM/model-merging
git reset --hard 3f83f02^  # Go back one commit before integration
```

Or to see changes:
```bash
git diff 3f83f02^  # Show all changes made during integration
```

## Files Changed Summary

**New files** (2):
- `src/model_merging/alignment/__init__.py`
- `src/model_merging/alignment/rotation_alignment.py`

**Modified files** (1):
- `scripts/evaluate_multitask_merging.py` (2 changes: import + alignment block)

**Configuration**:
- `conf/multitask.yaml` (already had `alignment: false`)

## Implementation Details

The rotation alignment uses the corrected SVD formula from the paper:
```
M = W_Q_local @ W_Q_anchor^T + W_K_local @ W_K_anchor^T +
    b_Q_local^T @ b_Q_anchor + b_K_local^T @ b_K_anchor
```

Then applies:
- `R = U @ V^T` where `U, V` from SVD(M)
- Aligned Q = `P.T @ Q_local` where `P = (V @ U.T).T`
- Aligned K = `P.T @ K_local`

This ensures:
1. Output invariance (model behavior unchanged)
2. Parameter alignment (weights move closer to anchor)

## Next Steps

1. **Test with alignment=false** to ensure backward compatibility
2. **Test with alignment=true** to verify rotation alignment works
3. **Compare merging results** with and without alignment
4. **Analyze if alignment improves multi-task merging performance**
