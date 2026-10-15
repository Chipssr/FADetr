import torch
import torch.nn as nn


class AngleGuidedAlignment(nn.Module):
    """
    Angle Guided Query Alignment (AGA)

    query : [B, N, C]
    theta : [B, N]

    output:
        aligned query [B, N, C]
    """

    def __init__(
        self,
        embed_dims=256,
        hidden_dims=64,
        scale=0.1,
    ):
        super().__init__()

        self.scale = scale

        self.angle_mlp = nn.Sequential(
            nn.Linear(2, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, embed_dims),
            nn.Sigmoid()
        )

    def forward(self, query, theta):

        angle_embed = torch.stack(
            [
                torch.sin(theta),
                torch.cos(theta)
            ],
            dim=-1
        )

        gate = self.angle_mlp(angle_embed)

        aligned_query = query + self.scale * query * gate

        return aligned_query
