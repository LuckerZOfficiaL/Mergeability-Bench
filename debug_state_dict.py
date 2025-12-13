"""
Debug script to check state_dict keys before and after alignment.
"""

import sys
sys.path.insert(0, 'src')

import torch
from model_merging.utils.io_utils import load_model_from_hf

# Load a model
print("Loading MNIST model...")
mnist_model = load_model_from_hf('ViT-B-16', 'MNIST')
state_dict = mnist_model.state_dict()

print(f"\nTotal keys: {len(state_dict)}")
print(f"\nFirst 10 keys:")
for i, key in enumerate(list(state_dict.keys())[:10]):
    print(f"  {i+1}. {key}")

# Check if they have 'model.' prefix
has_model_prefix = any(k.startswith('model.') for k in state_dict.keys())
print(f"\nHas 'model.' prefix: {has_model_prefix}")

# Count keys with and without prefix
model_keys = [k for k in state_dict.keys() if k.startswith('model.')]
other_keys = [k for k in state_dict.keys() if not k.startswith('model.')]

print(f"\nKeys with 'model.' prefix: {len(model_keys)}")
print(f"Keys without 'model.' prefix: {len(other_keys)}")

if other_keys:
    print(f"\nKeys without prefix:")
    for k in other_keys[:10]:
        print(f"  - {k}")

# Check for positional_embedding
pos_emb_keys = [k for k in state_dict.keys() if 'positional_embedding' in k]
print(f"\nPositional embedding keys: {pos_emb_keys}")
