import torch
import torch.nn as nn
import torch.nn.functional as F
from tinySwin import TinySwinTransformer  # your tiny swin wrapper

class UNet3DWithSwinFusion(nn.Module):
    def __init__(self, base_unet3d, out_channels, fusion_dim=512, swin_embed_dim=96, mode="projection"):
        super(UNet3DWithSwinFusion, self).__init__()
        self.unet3d = base_unet3d
        self.out_channels = out_channels
        self.fusion_dim = fusion_dim
        self.mode = mode  # 'projection' or 'attention'

        # Tiny Swin Transformer for extracting global context
        self.swin_transformer = TinySwinTransformer(
            input_dim=base_unet3d.final_conv.out_channels,  # matches UNet3D output channels
            embed_dim=swin_embed_dim
        )

        # Project swin_features to spatial size or channels
        self.projector = None  # create dynamically

        # Fusion conv after concatenation
        self.fusion_conv = None  # create dynamically

        # Final segmentation head
        self.final_conv = nn.Conv3d(base_unet3d.final_conv.out_channels, out_channels, kernel_size=1)
        self.final_activation = nn.Softmax(dim=1)  # or nn.Sigmoid() for binary segmentation

    def forward(self, x):
        B = x.shape[0]

        # Step 1: Send same input x to both UNet3D and TinySwin
        x_unet, _ = self.unet3d(x)  # (B, C, D, H, W)
        swin_features = self.swin_transformer(x)  # <-- x, not x_unet

        C, D, H, W = x_unet.shape[1], x_unet.shape[2], x_unet.shape[3], x_unet.shape[4]

        if self.mode == "projection":
            # --- Projection enhancement ---
            if self.projector is None:
                self.projector = nn.Linear(swin_features.shape[1], D * H * W)
                self.projector = self.projector.to(swin_features.device)

            enhancement = self.projector(swin_features)  # (B, D*H*W)
            enhancement = enhancement.view(B, 1, D, H, W)  # (B, 1, D, H, W)

            combined = torch.cat([x_unet, enhancement], dim=1)  # (B, C+1, D, H, W)
            '''
            if self.fusion_conv is None:
                self.fusion_conv = nn.Conv3d(
                    in_channels=combined.shape[1],  # (C+1)
                    out_channels=C,  # back to C
                    kernel_size=3,
                    padding=1
                )
                self.fusion_conv = self.fusion_conv.to(combined.device)
            '''
            #x_fused = self.fusion_conv(combined)
            x_fused = 0.8*x_unet+0.2*enhancement

        elif self.mode == "attention":
            # --- Attention modulation ---
            if self.projector is None:
                self.projector = nn.Linear(swin_features.shape[1], C)
                self.projector = self.projector.to(swin_features.device)

            attention = self.projector(swin_features)  # (B, C)
            attention = attention.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)
            attention = attention.expand(-1, -1, D, H, W)  # (B, C, D, H, W)
            attention = torch.sigmoid(attention)

            x_fused = x_unet * attention  # elementwise modulation

        else:
            raise ValueError(f"Unknown mode {self.mode}. Use 'projection' or 'attention'.")

        # Step 7: Final 1x1x1 convolution for segmentation
        out = self.final_conv(x_fused)

        if not self.training:
            out = self.final_activation(out)

        return out
