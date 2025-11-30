"""
Permutation specification for OpenCLIP ViT models.

OpenCLIP ViT parameter naming convention (for visual encoder):
- conv1.weight: Initial patch embedding convolution
- ln_pre.weight/bias: Layer norm before transformer
- transformer.resblocks.{i}.ln_1.weight/bias: Layer norm 1 in block i
- transformer.resblocks.{i}.attn.in_proj_weight/bias: Attention input projection (Q, K, V combined)
- transformer.resblocks.{i}.attn.out_proj.weight/bias: Attention output projection
- transformer.resblocks.{i}.ln_2.weight/bias: Layer norm 2 in block i
- transformer.resblocks.{i}.mlp.c_fc.weight/bias: MLP first layer
- transformer.resblocks.{i}.mlp.c_proj.weight/bias: MLP second layer
- ln_post.weight/bias: Final layer norm
- proj: Final projection (if present)
"""

from typing import Dict
from model_merging.permutations.permutation_spec import PermutationSpec, PermutationSpecBuilder


class CLIPViTPermutationSpecBuilder(PermutationSpecBuilder):
    """
    Permutation specification builder for OpenCLIP Vision Transformer models.

    Note: This handles the visual encoder portion of CLIP models (ViT-B-32, ViT-B-16, ViT-L-14, etc.)
    """

    def __init__(self, depth: int = 12, prefix: str = ""):
        """
        Args:
            depth: Number of transformer blocks (12 for Base, 24 for Large)
            prefix: Prefix for parameter names (e.g., "visual." if part of full CLIP model)
        """
        self.depth = depth
        self.prefix = prefix
        super().__init__()

    def _param(self, name: str) -> str:
        """Add prefix to parameter name if specified."""
        return f"{self.prefix}{name}" if self.prefix else name

    def create_permutation_spec(self, **kwargs) -> PermutationSpec:
        """
        Create permutation specification for OpenCLIP ViT.

        Key insight: In ViT, we can permute the hidden dimensions between blocks.
        Each block's MLP and attention create permutation matrices.
        """
        axes_to_perm = {}

        # Initial patch embedding - only output dimension can be permuted
        axes_to_perm[self._param("conv1.weight")] = ("P_embed", None, None, None)

        # Pre-transformer layer norm - permuted by embedding
        axes_to_perm[self._param("ln_pre.weight")] = ("P_embed",)
        axes_to_perm[self._param("ln_pre.bias")] = ("P_embed",)

        # Optional: position embedding and class token
        # These are typically not permuted, but if present:
        # axes_to_perm[self._param("positional_embedding")] = (None, "P_embed")
        # axes_to_perm[self._param("class_embedding")] = ("P_embed",)

        # Transformer blocks
        for i in range(self.depth):
            # Determine input and output permutations for this block
            p_in = "P_embed" if i == 0 else f"P_block{i-1}_out"
            p_out = f"P_block{i}_out" if i < self.depth - 1 else "P_final"
            p_mlp = f"P_block{i}_mlp"

            # Layer norm 1 (before attention) - uses input permutation
            axes_to_perm[self._param(f"transformer.resblocks.{i}.ln_1.weight")] = (p_in,)
            axes_to_perm[self._param(f"transformer.resblocks.{i}.ln_1.bias")] = (p_in,)

            # Attention module
            # in_proj combines Q, K, V - shape (3*dim, dim)
            # We permute the input dimension but not the output (Q, K, V are concatenated)
            axes_to_perm[self._param(f"transformer.resblocks.{i}.attn.in_proj_weight")] = (None, p_in)
            axes_to_perm[self._param(f"transformer.resblocks.{i}.attn.in_proj_bias")] = (None,)

            # out_proj - we don't permute attention output, only input
            # This creates the block output permutation
            axes_to_perm[self._param(f"transformer.resblocks.{i}.attn.out_proj.weight")] = (None, None)
            axes_to_perm[self._param(f"transformer.resblocks.{i}.attn.out_proj.bias")] = (None,)

            # Layer norm 2 (before MLP) - uses block input (residual connection)
            axes_to_perm[self._param(f"transformer.resblocks.{i}.ln_2.weight")] = (None,)
            axes_to_perm[self._param(f"transformer.resblocks.{i}.ln_2.bias")] = (None,)

            # MLP first layer (c_fc) - creates internal permutation
            axes_to_perm[self._param(f"transformer.resblocks.{i}.mlp.c_fc.weight")] = (p_mlp, None)
            axes_to_perm[self._param(f"transformer.resblocks.{i}.mlp.c_fc.bias")] = (p_mlp,)

            # MLP second layer (c_proj) - outputs to block output
            axes_to_perm[self._param(f"transformer.resblocks.{i}.mlp.c_proj.weight")] = (p_out, p_mlp)
            axes_to_perm[self._param(f"transformer.resblocks.{i}.mlp.c_proj.bias")] = (p_out,)

        # Post-transformer layer norm
        axes_to_perm[self._param("ln_post.weight")] = ("P_final",)
        axes_to_perm[self._param("ln_post.bias")] = ("P_final",)

        # Final projection (if present) - don't permute output (embedding space)
        # axes_to_perm[self._param("proj")] = (None, "P_final")

        return self.permutation_spec_from_axes_to_perm(axes_to_perm)


def create_clip_vit_spec(model_name: str, prefix: str = "model.visual.") -> PermutationSpec:
    """
    Convenience function to create CLIP ViT permutation spec based on model name.

    Args:
        model_name: Model name like "ViT-B-32", "ViT-B-16", "ViT-L-14", etc.
        prefix: Prefix for parameter names (default: "model.visual." for full CLIP models)

    Returns:
        PermutationSpec for the model
    """
    model_name_lower = model_name.lower()

    # Determine depth based on model variant
    if "vit-b" in model_name_lower or "vitb" in model_name_lower:
        depth = 12  # Base model has 12 transformer blocks
    elif "vit-l" in model_name_lower or "vitl" in model_name_lower:
        depth = 24  # Large model has 24 transformer blocks
    elif "vit-s" in model_name_lower or "vits" in model_name_lower:
        depth = 12  # Small model has 12 transformer blocks
    else:
        depth = 12  # Default to Base

    builder = CLIPViTPermutationSpecBuilder(depth=depth, prefix=prefix)
    return builder.create_permutation_spec()
