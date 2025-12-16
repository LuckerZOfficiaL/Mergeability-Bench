"""Quick script to check the actual layer names in the encoder model."""

from model_merging.utils.io_utils import load_model_from_hf

# Load pretrained encoder
pretrained_encoder = load_model_from_hf(model_name="ViT-B-16")

print("\n=== All module names in pretrained_encoder ===")
for name, module in pretrained_encoder.named_modules():
    if 'resblocks' in name or 'transformer' in name:
        print(f"{name}: {type(module).__name__}")

print("\n=== Checking for layer 11 specifically ===")
for name, module in pretrained_encoder.named_modules():
    if '11' in name and ('resblock' in name or 'layer' in name):
        print(f"{name}: {type(module).__name__}")
