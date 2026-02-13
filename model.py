import math
import torch
import torch.nn as nn

import numpy as np

import sys
sys.path.append("MS_diffusion/src")

from models.transformer_model import GraphTransformer_embedding

class TransformerModel(nn.Module):
    def __init__(self,
                 dim_sos=13,
                 dim_formula=144,
                 hidden_dim=256,
                 num_transformer_layers=4,
                 nhead=8,
                 output_dim=512,
                 dropout=0.1,
                 input_dropout = 0.1,
                 max_len=129):
        """
        Args:
            dim_sos (int): Input dimension for the sos token.
            dim_formula (int): Input dimension for each formula token.
            hidden_dim (int): The dimension to project tokens to.
            num_transformer_layers (int): Number of transformer encoder layers.
            nhead (int): Number of attention heads.
            mlp_hidden_dim (int): Hidden dimension in the MLP.
            dropout (float): Dropout rate used in positional encoding and transformer.
        """
        super(TransformerModel, self).__init__()

        # Linear projections to go from input dimensions to hidden_dim.
        self.sos_proj = nn.Linear(dim_sos, hidden_dim)
        self.formula_proj = nn.Linear(dim_formula, hidden_dim)
        
        # Sinusoidal positional encoding.
        self.input_dropout = nn.Dropout(p=input_dropout)
        
        # Transformer encoder definition.
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers, enable_nested_tensor=False)
        
        # A 2-layer MLP to process the first token after the transformer.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2048)
        )
        
        # Final layer normalization.
        self.norm = nn.LayerNorm(hidden_dim)

        scale = hidden_dim ** -0.5
        self.proj = nn.Parameter(scale * torch.randn(hidden_dim, output_dim))

        
    def forward(self, sos, formula_array, mask=None, fp = True):
        """
        Args:
            sos: Tensor of shape (B, 1, 13) containing the first token.
            formula_array: Tensor of shape (B, 128, 144) containing the rest of the tokens.
            mask: Optional boolean Tensor of shape (B, 129) where True indicates positions to ignore.
                  This will be passed as src_key_padding_mask to the transformer.
        Returns:
            Tensor of shape (B, hidden_dim) after processing.
        """
        # Project the inputs to hidden_dim.
        sos_proj = self.sos_proj(sos)  # (B, 1, d)
        formula_proj = self.formula_proj(formula_array)  # (B, 128, d)
        
        # Concatenate along the token dimension: (B, 129, d)
        tokens = torch.cat([sos_proj, formula_proj], dim=1)
        
        # Transformer encoder expects shape (seq_len, batch, hidden_dim).
        tokens = tokens.transpose(0, 1)  # (129, B, d)
        
        # # Add sinusoidal positional encoding.
        tokens = self.input_dropout(tokens)
        
        # Pass through the transformer encoder.
        transformer_out = self.transformer_encoder(tokens, src_key_padding_mask=mask)  # (129, B, d),  (B, 129)
        
        # Transpose back to (B, 129, 512).
        transformer_out = transformer_out.transpose(0, 1)
        
        # Take the first token (sos) from each batch element.
        first_token = transformer_out[:, 0, :]  # (B, d)
        output = self.norm (first_token) @ self.proj
        
        return output


class Contrastive_model(nn.Module):
    def __init__(self,
                 hidden_dim=256,
                 max_len = 129,
                 num_transformer_layers=4,
                 nhead=8,
                 embeddings_dim=512,
                 dropout=0.1,
                 input_dropout = 0.1,
                 fp_length = 2048,
                 graph = False,
                 fp_pred = False,
                 initial_temperature = 30.0,
                 trainable_temperature = False):    
        """
        Args:
            dim_sos (int): Input dimension for the sos token.
            dim_formula (int): Input dimension for each formula token.
            hidden_dim (int): The dimension to project tokens to.
            num_transformer_layers (int): Number of transformer encoder layers.
            nhead (int): Number of attention heads.
            mlp_hidden_dim (int): Hidden dimension in the MLP.
            dropout (float): Dropout rate used in positional encoding and transformer.
        """
        super(Contrastive_model, self).__init__()

        self.ms_encoder = TransformerModel(
                hidden_dim=hidden_dim,
                max_len = max_len,
                num_transformer_layers=num_transformer_layers,
                nhead=nhead,
                output_dim=embeddings_dim,
                dropout=dropout,
                input_dropout = input_dropout)

        self.graph = graph
        self.fp_pred = fp_pred

        if self.graph:
            self.graph_encoder = GraphTransformer_embedding(n_layers=4,
                                input_dims={'X': 11, 'E': 5, 'y':1},
                                hidden_mlp_dims={'X': 256, 'E': 128, 'y': 1},
                                hidden_dims={'dx': 256, 'de': 128, 'dy': 1, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 1},
                                output_dims={'X': embeddings_dim},
                                act_fn_in=nn.ReLU(),
                                act_fn_out=nn.ReLU())

        if self.fp_pred:
            self.out_mlp = nn.Sequential(
            nn.Linear(embeddings_dim, embeddings_dim),
            nn.ReLU(),
            nn.Linear(embeddings_dim, fp_length))
        
        if trainable_temperature:
            self.inv_temperature = nn.Parameter(torch.tensor(1.0/initial_temperature))

    
    def forward(self, sos, formula_array, fp=None, graph = None, mask=None):

        ms_embeddings = self.ms_encoder(sos, formula_array, mask).float()

        if self.graph:
            graph_embeddings = self.graph_encoder(graph.X, graph.E, graph.y, graph.node_mask)

            if self.fp_pred:
                fp_output = self.out_mlp(ms_embeddings)
                return ms_embeddings, graph_embeddings, fp_output
            
            else:
                return ms_embeddings, graph_embeddings

        elif self.fp_pred:
            fp_output = self.out_mlp(ms_embeddings)
            return fp_output

        else:
            return ms_embeddings
