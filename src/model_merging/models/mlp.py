"""
MLP model for predicting mergeability scores.
"""
import torch
import torch.nn as nn


class MergeabilityMLP(nn.Module):
    """
    Two-layer MLP for predicting mergeability scores for multiple merge methods.

    Architecture:
        Input (23) -> Hidden (48) -> Dropout -> Output (4)

    Where:
        - Input: 23-dimensional vector of mergeability metrics
        - Hidden: 48-dimensional hidden layer with LeakyReLU activation
        - Dropout: Regularization layer
        - Output: 4-dimensional vector (one score per merge method)
    """

    def __init__(self, input_dim=23, hidden_dim=48, output_dim=4, dropout=0.2):
        """
        Initialize MLP.

        Args:
            input_dim: Number of input features (mergeability metrics)
            hidden_dim: Number of hidden units
            output_dim: Number of output dimensions (merge methods)
            dropout: Dropout probability for regularization
        """
        super(MergeabilityMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights with Kaiming initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch_size, input_dim) tensor of input features

        Returns:
            (batch_size, output_dim) tensor of predicted mergeability scores
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)  # Apply dropout after activation
        x = self.fc2(x)
        return x

    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
