import torch
import torch.nn as nn
import torch.nn.functional as F
from tinySwin import TinySwinTransformer  # your tiny swin wrapper

class UNet3DWithSwinGatedFusion(nn.Module):
    def __init__(self, base_unet3d, out_channels, swin_embed_dim=96):
        super(UNet3DWithSwinGatedFusion, self).__init__()
        self.unet3d = base_unet3d
        self.out_channels = out_channels

        # Tiny Swin Transformer for extracting global context
        self.swin_transformer = TinySwinTransformer(
            input_dim=base_unet3d.final_conv.out_channels,
            embed_dim=swin_embed_dim
        )

        # Project Swin features to match UNet channel size (for gating)
        self.channel_gate = None  # will initialize dynamically

        # Final segmentation head
        self.final_conv = nn.Conv3d(
            base_unet3d.final_conv.out_channels, out_channels, kernel_size=1
        )
        self.final_activation = nn.Softmax(dim=1)  # or Sigmoid for binary

    def forward(self, x):
        B = x.shape[0]

        # --- UNet3D Backbone ---
        x_unet, _ = self.unet3d(x)  # (B, C, D, H, W)
        C, D, H, W = x_unet.shape[1:]

        # --- Swin Transformer Path ---
        swin_features = self.swin_transformer(x)  # (B, E)

        # --- Gated Fusion ---
        if self.channel_gate is None:
            self.channel_gate = nn.Linear(swin_features.shape[1], C).to(x.device)

        gate = torch.sigmoid(self.channel_gate(swin_features))  # (B, C)
        gate = gate.view(B, C, 1, 1, 1)  # broadcastable shape
        x_fused = x_unet * gate  # Element-wise modulation

        # --- Final Prediction ---
        out = self.final_conv(x_fused)

        if not self.training:
            out = self.final_activation(out)

        return out
