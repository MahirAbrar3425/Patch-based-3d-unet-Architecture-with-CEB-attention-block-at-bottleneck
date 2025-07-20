# PyTorch core
import torch
import torch.nn as nn
import torch.nn.functional as F

# Swin Transformer model
from timm.models.swin_transformer import SwinTransformer
from tinySwin import TinySwinTransformer

# TinySwinTransformer is your custom class — make sure it's defined
# Example:
# from your_module import TinySwinTransformer

class UNet3DWithSwinFusion(nn.Module):
    def __init__(self, base_unet3d, out_channels, fusion_dim=512, swin_embed_dim=96):
        super(UNet3DWithSwinFusion, self).__init__()
        self.unet3d = base_unet3d
        self.out_channels = out_channels
        self.fusion_dim = fusion_dim

        # Tiny Swin Transformer for feature extraction
        self.swin_transformer = TinySwinTransformer(input_dim=base_unet3d.final_conv.out_channels, embed_dim=swin_embed_dim)

        # Fusion head
        self.fusion_head = None

        self.final_activation = nn.Softmax(dim=1)  # or Sigmoid if binary

    def forward(self, x):
        # Step 1: UNet3D output
        x_unet, _ = self.unet3d(x)  # (B, C, D, H, W)

        # Step 2: Swin Transformer output
        swin_features = self.swin_transformer(x_unet)  # (B, swin_embed_dim)

        # Step 3: Global pool UNet3D output
        x_unet_pool = torch.mean(x_unet, dim=[2, 3, 4])  # (B, out_channels)

        # Step 4: Fuse features
        fused = torch.cat([x_unet_pool, swin_features], dim=1)  # (B, out_channels + swin_embed_dim)

        # Step 5: Create Fusion Head dynamically if not initialized
        if self.fusion_head is None:
            self.fusion_head = nn.Sequential(
                nn.Linear(fused.shape[1], self.fusion_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.fusion_dim, self.out_channels)
            )
        self.fusion_head = self.fusion_head.to(fused.device)

        # Step 6: Predict
        pred = self.fusion_head(fused)
        pred = self.final_activation(pred)

        return pred