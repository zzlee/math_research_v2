import torch
import torch.nn as nn
from torch.nn import functional as F
import random

# =========================================
# 1. Architecture for Model V2
# =========================================
D_MODEL = 128
N_LAYER = 3
VOCAB_SIZE = 15
BLOCK_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_embedding = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.layers = nn.ModuleList([
            nn.Sequential(
                # This is a simplification; we need to match the state_dict keys exactly
                # The keys are layers.N.mha..., layers.N.ff..., layers.N.norm...
            ) for _ in range(N_LAYER)
        ])
        self.fc_out = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, idx):
        # Not used for just loading and testing generation if we use a manual loop
        pass

# Because the model structure in the state_dict is specific (mha.in_proj, etc.),
# we'll define a more precise architecture to match the keys.
class MHALayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 3 * d_model) # q, k, v
        self.out_proj = nn.Linear(3 * d_model // 3, d_model) # Assuming 1 head for simplicity in load, or matching the shape

class FFLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

class BlockV2(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mha = nn.Module() 
        # We will manually assign weights to avoid architecture mismatch
        self.ff = FFLayer(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

# Since loading a complex state_dict into a custom class is error-prone,
# and the previous subagent failed, I will use a "Weight-Driven" Generation loop
# or a simplified torch.nn.Module that matches the state_dict keys exactly.

class TransformerV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_embedding = nn.Parameter(torch.randn(1, BLOCK_SIZE, D_MODEL))
        
        self.layers = nn.ModuleList()
        for _ in range(N_LAYER):
            layer = nn.Module()
            # MHA components
            layer.mha = nn.Module()
            layer.mha.in_proj = nn.Linear(D_MODEL, 3 * D_MODEL)
            layer.mha.out_proj = nn.Linear(D_MODEL, D_MODEL)
            # FF components
            layer.ff = nn.Sequential(
                nn.Linear(D_MODEL, 4 * D_MODEL),
                nn.ReLU(),
                nn.Linear(4 * D_MODEL, D_MODEL)
            )
            layer.norm1 = nn.LayerNorm(D_MODEL)
            layer.norm2 = nn.LayerNorm(D_MODEL)
            self.layers.append(layer)
            
        self.fc_out = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, idx):
        B, T = idx.shape
        x = self.embedding(idx) + self.pos_embedding[:, :T, :]
        
        for layer in self.layers:
            # Simplified MHA for verification
            # x = x + layer.mha(...) # complex to implement from scratch here
            # To make this actually work, we should implement the actual MHA forward.
            pass
        
        return self.fc_out(x)

# Actually, instead of rewriting the whole transformer, I'll just use the state_dict 
# to verify if it can generate basic tokens. 
# But for accuracy, I need a working forward pass.

# Let's look at the original train_math_cot.py and adapt it for V2.
