"""Test to check the actual format of state dicts from load_model_from_hf"""
import sys
sys.path.insert(0, 'src')

from model_merging.utils.io_utils import load_model_from_hf

# Load MNIST model
print("Loading MNIST model...")
model = load_model_from_hf(model_name="ViT-B-16", dataset_name="MNIST")

print("\nGetting state_dict...")
state_dict = model.state_dict()

print(f"\nTotal keys: {len(state_dict)}")
print(f"\nFirst 10 keys:")
for i, key in enumerate(list(state_dict.keys())[:10]):
    print(f"  {i+1}. {key}")

print(f"\nHas 'model.' prefix: {any(k.startswith('model.') for k in state_dict.keys())}")
print(f"Has 'visual.' prefix: {any(k.startswith('visual.') for k in state_dict.keys())}")
print(f"Has 'model.visual.' prefix: {any(k.startswith('model.visual.') for k in state_dict.keys())}")
