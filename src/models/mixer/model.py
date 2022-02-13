from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath


class MlpBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, c, num_patch, tokens_mlp_dim, channel_mlp_dim, drop_path_rate):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(c),

            # n -> num_patches, d -> `c` hidden dimension
            Rearrange('b n c -> b c n'),
            MlpBlock(num_patch, tokens_mlp_dim),
            Rearrange('b c n -> b n c'),
        )
        self.drop_path = drop_path_rate

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(c),
            MlpBlock(c, channel_mlp_dim)
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)

        return x


class MlpMixer(nn.Module):
    def __init__(self, in_channels, c, num_classes, patch_size, image_size,
                 depth, token_mlp_dim, channel_mlp_dim, drop_path_max_rate=0.1):
        super().__init__()

        assert image_size % patch_size == 0, "Patch size should divide image size"
        num_patches = (image_size // patch_size) ** 2

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c,
                      kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(c, num_patches, token_mlp_dim, channel_mlp_dim, 0 + i * drop_path_max_rate / (depth - 1))
             for i in range(depth)])

        self.layer_norm = nn.LayerNorm(c)

        self.final_fc = nn.Sequential(
            nn.Linear(c, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)

        for block in self.mixer_blocks:
            x = block(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)

        x = self.final_fc(x)
        return x
