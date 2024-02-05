from .vistabnet_backbone import Encoder
from torchvision.models.vision_transformer import VisionTransformer, ViT_B_16_Weights
from torch import nn
from functools import partial

import torch

from functools import partial

class TabularVisionTransformer(torch.nn.Module):
    class TabularVit(VisionTransformer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.embed_dim = kwargs["hidden_dim"]
            self.encoder = Encoder(
                197,
                kwargs["num_layers"],
                kwargs["num_heads"],
                kwargs["hidden_dim"],
                kwargs["mlp_dim"],
                0.0,
                0.0,
                partial(nn.LayerNorm, eps=1e-6),
            )

            weights = ViT_B_16_Weights.DEFAULT.get_state_dict(progress=True)
            self.load_state_dict(weights)

        def _preprocess_input(self, x):
            return x

        def forward(self, x, mask=None):
            b, emb, f = x.shape
            x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1)
            if hasattr(self, 'positional_embedding'): 
                x = self.positional_embedding(x)

            if mask is not None:
                mask = torch.cat((torch.ones(1, 1).to(x.device), mask), dim=1).to(torch.bool)
                
            x = self.encoder(x, mask)
            if hasattr(self, 'pre_logits'):
                x = self.pre_logits(x)
                x = torch.tanh(x)
            return x
        
    class Projector(torch.nn.Module):
        # 196 projections which will be computed in parallel
        def __init__(self, input_size, projections, proj_depth, output_size):
            super().__init__()
            self.projections = projections

            self.projector = torch.nn.Parameter(torch.rand(self.projections, input_size, output_size))
            self.bias = torch.nn.Parameter(torch.rand(self.projections, output_size))

            # self.projector_hidden = torch.nn.Linear(output_size, output_size)
            self.projector_hidden = []
            for i in range(proj_depth): 
                self.projector_hidden.append(torch.nn.Linear(output_size, output_size))
                self.projector_hidden.append(torch.nn.GELU())
                self.projector_hidden.append(torch.nn.LayerNorm(output_size, eps=1e-6))

            self.projector_hidden = torch.nn.Sequential(*self.projector_hidden)

            self.activation = torch.nn.GELU()
            self.layernorm = torch.nn.LayerNorm(output_size, eps=1e-6)
            # self.batchnorm = torch.nn.BatchNorm1d(output_size)

        def forward(self, x):
            #input shape: (batch_size, input_size)
            #output shape: (batch_size, 196, output_size)
            x = x.unsqueeze(1).repeat(1, self.projections, 1)
            x = torch.einsum("bei,eio->beo", x, self.projector)
            x = x + self.bias
            x = self.activation(x)
            x = self.layernorm(x)

            x = self.projector_hidden(x)

            # x = torch.einsum("bei,eio->beo", x, self.projector_hidden)
            # x = x + self.hidden_bias
            # x = self.activation(x)
            return x
        
    class Head(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.head = torch.nn.Linear(input_size, 128)
            self.head_hidden = torch.nn.Linear(128, 128)
            self.head2 = torch.nn.Linear(128, output_size)
            self.softmax = torch.nn.Softmax(dim=1)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.head(x)
            x = self.relu(x)
            x = self.head_hidden(x)
            x = self.relu(x)
            x = self.head2(x)
            x = self.softmax(x)
            return x
        
    def __init__(self, input_size, output_size, projections, proj_depth, *args, **kwargs):
        super().__init__()
        self.vit = self.TabularVit(*args, **kwargs)
        self.projector = self.Projector(input_size, projections, proj_depth, self.vit.embed_dim)
        self.head = self.Head(self.vit.embed_dim, output_size)

        self.vit.requires_grad_(False)

    def forward(self, x):
        x = self.projector(x)
        mask = torch.ones(1, x.shape[1]).to(x.device)
        mask = torch.cat((mask, torch.zeros(1, 196 - x.shape[1]).to(x.device)), dim=1)

        padding = torch.zeros(x.shape[0], 196 - x.shape[1], x.shape[2]).to(x.device)
        x = torch.cat((x, padding), dim=1)

        x = self.vit(x, mask)[:, 0, :]
        x = self.head(x)
        return x