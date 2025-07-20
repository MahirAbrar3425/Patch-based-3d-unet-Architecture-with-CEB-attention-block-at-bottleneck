import torch
import torch.nn as nn
import torch.nn.functional as F
from tinySwin import TinySwinTransformer  # your tiny swin wrapper

class UNet3DWithSwinFusion(nn.Module):
    def __init__(self, base_unet3d, out_channels, fusion_dim=512, swin_embed_dim=96):
        super(UNet3DWithSwinFusion, self).__init__()
        self.unet3d = base_unet3d
        self.out_channels = out_channels
        self.fusion_dim = fusion_dim

        # Tiny Swin Transformer for extracting global context
        self.swin_transformer = TinySwinTransformer(
            input_dim=base_unet3d.final_conv.out_channels,
            embed_dim=swin_embed_dim
        )

        # New enhancement MLP
        self.enhance_head = None

        self.fusion_conv = None  # create dynamically

        self.final_conv = nn.Conv3d(base_unet3d.final_conv.out_channels, out_channels, kernel_size=1)
        self.final_activation = nn.Softmax(dim=1)  # or nn.Sigmoid() for binary segmentation

    def forward(self, x):
        # Step 1: UNet3D backbone output
        x_unet, _ = self.unet3d(x)  # (B, C, D, H, W)

        # Step 2: Extract Swin global features
        swin_features = self.swin_transformer(x_unet)  # (B, swin_embed_dim)

        # Step 3: Global pool UNet feature map
        x_unet_pool = torch.mean(x_unet, dim=[2, 3, 4])  # (B, C)

        # Step 4: Fuse pooled UNet and Swin features
        fused = torch.cat([x_unet_pool, swin_features], dim=1)  # (B, C + swin_embed_dim)

        # Step 5: Enhancement
            # Step 5: Dynamically create enhance_head
        if self.enhance_head is None:
            self.enhance_head = nn.Sequential(
                nn.Linear(fused.shape[1], self.fusion_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.fusion_dim, x_unet.shape[1])  # output C same as UNet features
            )
        self.enhance_head = self.enhance_head.to(fused.device)
        enhancement = self.enhance_head(fused)  # (B, C)



        # Step 6: Expand enhancement back to 3D
        enhancement = enhancement.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)
        enhancement = enhancement.expand_as(x_unet)  # (B, C, D, H, W)

        # Step 7: Combine original UNet output and enhancement
        #x_fused = x_unet + enhancement  # simple residual addition
        combined = torch.cat([x_unet, enhancement], dim=1)  # (B, 2C, D, H, W)

        # Step 8: Create fusion_conv dynamically
        if self.fusion_conv is None:
            self.fusion_conv = nn.Conv3d(
                in_channels=combined.shape[1],  # 2C
                out_channels=x_unet.shape[1],  # back to C
                kernel_size=3,
                padding=1
            )
            self.fusion_conv = self.fusion_conv.to(combined.device)

        x_fused = self.fusion_conv(combined)  # (B, C, D, H, W)


        # Step 9: Final 1x1x1 convolution to predict segmentation map
        out = self.final_conv(x_fused)

        # Step 10: During evaluation, apply activation
        if not self.training:
            out = self.final_activation(out)

        return out
