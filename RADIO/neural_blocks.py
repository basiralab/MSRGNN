from functools import partial
from typing import Callable, Optional

import torch
from torch import nn


class LinearBNReLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(LinearBNReLU, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        shape = x.shape
        x = x.flatten(0, -2)
        x = self.bn(x)
        x = x.view(shape)
        x = self.relu(x)
        return x


class NonLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm: str = "bn"):
        assert norm in ["bn", "ln", "none"]
        super(NonLinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if norm == "bn":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm == "ln":
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        shape = x.shape
        x = x.flatten(0, -2)
        x = self.norm(x)
        x = x.view(shape)
        x = self.relu(x)
        return x


class DeepLinearBNReLU(nn.Module):
    def __init__(
        self, depth: int, in_dim: int, out_dim: int, change_dim_first: bool = True
    ):
        super(DeepLinearBNReLU, self).__init__()
        layers = []
        if change_dim_first:
            layers += [LinearBNReLU(in_dim, out_dim)]
            for _ in range(depth - 1):
                layers += [LinearBNReLU(out_dim, out_dim)]
        else:
            for _ in range(depth - 1):
                layers += [LinearBNReLU(in_dim, in_dim)]
            layers += [LinearBNReLU(in_dim, out_dim)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MLP(nn.Module):
    def __init__(
        self,
        depth: int,
        in_dim: int,
        out_dim: int,
        change_dim_first: bool = True,
        norm: str = "bn",
    ):
        assert norm in ["bn", "ln", "none"]
        super(MLP, self).__init__()
        layers = []
        if change_dim_first:
            layers += [NonLinear(in_dim, out_dim, norm)]
            for _ in range(depth - 1):
                layers += [NonLinear(out_dim, out_dim, norm)]
        else:
            for _ in range(depth - 1):
                layers += [NonLinear(in_dim, in_dim, norm)]
            layers += [NonLinear(in_dim, out_dim, norm)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvBnRelu(nn.Module):
    def __init__(self, num_input_channels: int, num_output_channels: int, **kwargs):
        super(ConvBnRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_input_channels, num_output_channels, **kwargs),
            nn.BatchNorm2d(num_output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class FeedForwardResidualBlock(nn.Module):
    def __init__(self, dim: int, expansion_multiplier: int = 1):
        super(FeedForwardResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * expansion_multiplier),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dim * expansion_multiplier),
            nn.Linear(dim * expansion_multiplier, dim),
        )

    def forward(self, x: torch.Tensor):
        return x + self.layers(x)


def FeedForward(
    dim: int,
    expansion_factor: int = 4,
    dropout: float = 0.0,
    dense: Callable[..., nn.Module] = nn.Linear,
    activation: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
    output_dim: Optional[int] = None,
):
    output_dim = output_dim if output_dim else dim
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        activation(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, output_dim),
        nn.Dropout(dropout),
    )


class ResidualPreNormFeedForward(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, **kwargs)

    def forward(self, x):
        return self.ff(self.norm(x)) + x


class ResidualFeedForward(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.ff = FeedForward(dim, **kwargs)

    def forward(self, x):
        return self.ff(x) + x


class Scattering(nn.Module):
    def __init__(self, num_groups: int):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Equivalent to Rearrange('b c (ng gs) -> b ng (c gs)', ng=num_groups, gs=group_size)
        :param x: a Tensor with rank >= 3 and last dimension divisible by number of groups
        :param num_groups: number of groups
        """
        shape_1 = x.shape[:-1] + (self.num_groups,) + (x.shape[-1] // self.num_groups,)
        x = x.view(shape_1)
        x = x.transpose(-3, -2).contiguous()
        return x.flatten(start_dim=-2)


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class GroupObjectsIntoPairs(nn.Module):
    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects, object_size = objects.size()
        return torch.cat(
            [
                objects.unsqueeze(1).repeat(1, num_objects, 1, 1),
                objects.unsqueeze(2).repeat(1, 1, num_objects, 1),
            ],
            dim=3,
        ).view(batch_size, num_objects**2, 2 * object_size)


class Sum(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=self.dim)


class TagPanelEmbeddings(nn.Module):
    """Tags panel embeddings with their absolute coordinates."""

    def forward(
        self, panel_embeddings: torch.Tensor, num_context_panels: int
    ) -> torch.Tensor:
        """
        Concatenates a one-hot encoded vector to each panel.
        The concatenated vector indicates panel absolute position in the RPM.
        :param panel_embeddings: a tensor of shape (batch_size, num_panels, embedding_size)
        :return: a tensor of shape (batch_size, num_panels, embedding_size + 9)
        """
        batch_size, num_panels, _ = panel_embeddings.shape
        tags = torch.zeros((num_panels, 9), device=panel_embeddings.device).type_as(
            panel_embeddings
        )
        tags[:num_context_panels, :num_context_panels] = torch.eye(
            num_context_panels, device=panel_embeddings.device
        ).type_as(panel_embeddings)
        if num_panels > num_context_panels:
            tags[num_context_panels:, num_context_panels] = torch.ones(
                num_panels - num_context_panels, device=panel_embeddings.device
            ).type_as(panel_embeddings)
        tags = tags.expand((batch_size, -1, -1))
        return torch.cat([panel_embeddings, tags], dim=2)


def arrange_for_ravens_matrix(
    x: torch.Tensor, num_context_panels: int, num_answer_panels: int
) -> torch.Tensor:
    batch_size, num_panels, embedding_dim = x.shape
    x = torch.stack(
        [
            torch.cat((x[:, :num_context_panels], x[:, i].unsqueeze(1)), dim=1)
            for i in range(num_context_panels, num_panels)
        ],
        dim=1,
    )
    x = x.view(batch_size * num_answer_panels, num_context_panels + 1, embedding_dim)
    return x


def arrange_for_odd_one_out(x: torch.Tensor) -> torch.Tensor:
    batch_size, num_panels, embedding_dim = x.shape
    mask = (
        ~torch.eye(num_panels, device=x.device, dtype=torch.bool)
        .unsqueeze(-1)
        .repeat(1, 1, embedding_dim)
    )
    x = torch.stack(
        [
            x.masked_select(m.repeat(batch_size, 1, 1)).view(
                batch_size, num_panels - 1, embedding_dim
            )
            for m in mask
        ],
        dim=1,
    )  # b p p-1 d
    x = x.view((batch_size * num_panels), (num_panels - 1), embedding_dim)
    return x
