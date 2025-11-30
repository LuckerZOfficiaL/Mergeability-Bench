"""
Quick script to inspect CLIP model parameter names to help build permutation spec.
"""
import sys
sys.path.insert(0, '/home/ubuntu/thesis/MM/model-merging/src')

from model_merging.utils.io_utils import load_model_from_hf

# Load a CLIP model
model_name = "ViT-B-32"
print(f"Loading {model_name}...")
model = load_model_from_hf(model_name=model_name)

# Get state dict
state_dict = model.state_dict()

print(f"\nTotal parameters: {len(state_dict)}")
print("\nFirst 30 parameter names:")
for i, key in enumerate(sorted(state_dict.keys())[:30]):
    shape = state_dict[key].shape
    print(f"{i+1:3d}. {key:60s} {str(shape):30s}")

print("\n\nSearching for transformer blocks...")
transformer_params = [k for k in state_dict.keys() if 'resblock' in k.lower() or 'block' in k.lower()]
if transformer_params:
    print(f"Found {len(transformer_params)} transformer block parameters")
    print("Sample transformer block parameters:")
    for key in sorted(transformer_params)[:20]:
        print(f"  {key}")
