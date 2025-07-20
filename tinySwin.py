import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer
# -------------------------------


# -------------------------------
# TINY SWIN TRANSFORMER MODULE
# -------------------------------
class TinySwinTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=96, depths=(2, 2), num_heads=(3, 6), swin_downsample_size=64):
        super(TinySwinTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.swin = None  # delay initialization
        self.swin_downsample_size = swin_downsample_size  # downsample to this size if H/W is too big

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = torch.mean(x, dim=2)  # mean over depth -> (B, C, H, W)

        B, C, H, W = x.shape

        # Optional downsampling if size is large
        if max(H, W) > self.swin_downsample_size:
            x = F.interpolate(x, size=(self.swin_downsample_size, self.swin_downsample_size), mode='bilinear', align_corners=False)
            H, W = self.swin_downsample_size, self.swin_downsample_size

        if self.swin is None:
            # Create SwinTransformer dynamically
            self.swin = SwinTransformer(
                img_size=(H, W),
                patch_size=1,
                in_chans=self.input_dim,
                embed_dim=self.embed_dim,
                depths=self.depths,
                num_heads=self.num_heads,
                window_size=min(H, W),  # safe window size
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.,
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
                num_classes=0  # no classifier head
            )
            self.swin = self.swin.to(x.device)

        out = self.swin(x)  # output is (B, embed_dim)
        return out