"""
Quick script to inspect CLIP model parameter names to help build permutation spec.
"""
import open_clip
import torch

# Load a CLIP model
model_name = "ViT-B-32"
pretrained = "openai"

print(f"Loading {model_name} ({pretrained})...")
model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

# Get state dict - just the visual encoder
if hasattr(model, 'visual'):
    state_dict = model.visual.state_dict()
    print("Using model.visual")
else:
    state_dict = model.state_dict()
    print("Using full model")

print(f"\nTotal parameters: {len(state_dict)}")
print("\nAll parameter names (first 50):")
for i, key in enumerate(sorted(state_dict.keys())[:50]):
    shape = state_dict[key].shape
    print(f"{i+1:3d}. {key:70s} {str(shape):30s}")

print("\n\nSearching for transformer/attention blocks...")
block_params = [k for k in state_dict.keys() if 'resblock' in k.lower() or 'transformer' in k.lower()]
if block_params:
    print(f"\nFound {len(block_params)} block parameters")
    print("\nTransformer block structure (first block):")
    block_0_params = [k for k in sorted(block_params) if '.0.' in k or 'resblocks.0' in k]
    for key in block_0_params[:15]:
        print(f"  {key}")
