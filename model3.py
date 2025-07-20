import torch
import torch.nn as nn
import torchvision.models as models
#import torch.nn.functional as F
from td_attention_unet import UNet3D
from BuildingBlocks import FinalConv

class UNet3DWithViT(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=16, layer_order='crg', num_groups=8, **kwargs):
        super(UNet3DWithViT, self).__init__()

        # 3D U-Net initialization
        self.unet = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            **kwargs
        )

        # Load Vision Transformer
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove classification head
        self.final_conv = nn.Conv3d(16, out_channels, 1)

        # Define a linear layer to project ViT features
        dummy_input = torch.randn(1, 3, 224, 224)  # Example input
        vit_features = self.vit(dummy_input)
        vit_feature_dim = vit_features.shape[-1]
        self.vit_projection = nn.Linear(vit_feature_dim, f_maps)
        self.channel_projection = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
        self.final_activation = nn.Sigmoid() if final_sigmoid else nn.Softmax(dim=1)


    def forward(self, x):
        # Extract 3D U-Net outputs
        unet_output, pool_fea = self.unet(x)

        # Prepare input for ViT
        batch_size, channels, depth, height, width = x.size()
        vit_input = x.view(batch_size * depth, channels, height, width)  # Flatten depth
        if channels == 1:  # Duplicate channel if grayscale
            vit_input = vit_input.repeat(1, 3, 1, 1)
        if channels != 3:
            vit_input = self.channel_projection(vit_input)
        # Resize to match Vision Transformer input size
        vit_input = nn.functional.interpolate(vit_input, size=(224, 224), mode="bilinear", align_corners=False)

        vit_features = self.vit(vit_input)  # Pass through ViT
        vit_features = vit_features.view(batch_size, depth, -1).mean(dim=1)  # Aggregate across depth

        # Project ViT features
        vit_features = self.vit_projection(vit_features)

        # Combine U-Net and ViT features
        combined_features = pool_fea + vit_features
        combined_features = self.final_conv(combined_features)
        combined_features = self.final_activation(combined_features)

        return unet_output, combined_features
