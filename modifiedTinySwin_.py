import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformer


## this file is used for UNet3dWithSkipSwinPurifier


class TinySwinTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=96, depths=(2, 2), num_heads=(3, 6), swin_downsample_size=64):
        super(TinySwinTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.swin = None  # lazy init
        self.swin_downsample_size = swin_downsample_size
        self.pre_conv = nn.Conv2d(256, 64, kernel_size=1)  # Add in __init__

        # After Swin, map back to input_dim
        self.channel_mapper = None  # also lazy init



    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape

        # Collapse depth (simple 2D representation)
        x_2d = torch.mean(x, dim=2)  # (B, C, H, W)

        # Resize to fixed Swin input
        if max(H, W) > self.swin_downsample_size:
            x_2d = F.interpolate(x_2d, size=(self.swin_downsample_size, self.swin_downsample_size), mode='bilinear', align_corners=False)
            H_new, W_new = self.swin_downsample_size, self.swin_downsample_size
        else:
            H_new, W_new = H, W

        # Pre-conv to match Swin input channel count
        x_2d = self.pre_conv(x_2d)  # (B, 64, H_new, W_new)

        # Initialize Swin if not yet initialized
        if self.swin is None:
            self.swin = SwinTransformer(
                img_size=(H_new, W_new),
                patch_size=1,
                in_chans=64,  # Must match output of pre_conv
                embed_dim=self.embed_dim,
                depths=self.depths,
                num_heads=self.num_heads,
                window_size=min(H_new, W_new),
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.,
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
                num_classes=0
            ).to(x.device)

        # Flatten for Swin input
        x_flat = x_2d.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Forward through Swin
        swin_feats = self.swin.forward_features(x_flat)  # (B, N, embed_dim)
        swin_feats = swin_feats.transpose(1, 2).reshape(B, self.embed_dim, H_new, W_new)  # (B, embed_dim, H_new, W_new)

        # Lazy init channel mapper
        if self.channel_mapper is None:
            self.channel_mapper = nn.Conv2d(self.embed_dim, self.input_dim, kernel_size=1).to(x.device)

        # Map to original input_dim
        swin_feats = self.channel_mapper(swin_feats)  # (B, C, H_new, W_new)

        # Resize back to original H, W
        swin_feats = F.interpolate(swin_feats, size=(H, W), mode='bilinear', align_corners=False)

        # Expand to 3D
        swin_feats = swin_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, C, D, H, W)

        # Apply gating
        purified = x * torch.sigmoid(swin_feats)
        return purified
