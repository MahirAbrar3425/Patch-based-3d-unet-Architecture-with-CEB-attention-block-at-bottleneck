import torch
import torch.nn as nn
import timm
from td_attention_unet import UNet3D


class UNet3DWithSwinTiny(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=16, layer_order='crg', num_groups=8, **kwargs):
        super(UNet3DWithSwinTiny, self).__init__()

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

        # Load Swin-Tiny model
        self.swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
        self.swin.head = nn.Identity()  # Remove classification head

        # Define a linear layer to project Swin-Tiny features to match f_maps
        dummy_input = torch.randn(1, 3, 224, 224)  # Example input
        swin_features = self.swin(dummy_input)
        swin_feature_dim = swin_features.shape[-1]
        self.swin_projection = nn.Linear(swin_feature_dim, f_maps)

        # Channel projection for non-RGB input
        self.channel_projection = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1)
        self.final_conv = nn.Conv3d(f_maps, out_channels, 1)
        self.final_activation = nn.Sigmoid() if final_sigmoid else nn.Softmax(dim=1)

    def forward(self, x):
        # Extract 3D U-Net outputs
        unet_output, pool_fea = self.unet(x)

        # Prepare input for Swin-Tiny
        batch_size, channels, depth, height, width = x.size()
        swin_input = x.view(batch_size * depth, channels, height, width)  # Flatten depth
        if channels == 1:  # Duplicate channel if grayscale
            swin_input = swin_input.repeat(1, 3, 1, 1)
        if channels != 3:
            swin_input = self.channel_projection(swin_input)
        # Resize to match Swin-Tiny input size
        swin_input = nn.functional.interpolate(swin_input, size=(224, 224), mode="bilinear", align_corners=False)

        # Pass through Swin-Tiny
        swin_features = self.swin(swin_input)  # (B*D, C)
        swin_features = swin_features.view(batch_size, depth, -1).mean(dim=1)  # Aggregate across depth

        # Project Swin-Tiny features
        swin_features = self.swin_projection(swin_features)

        # Combine U-Net and Swin-Tiny features
        combined_features = pool_fea + swin_features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        combined_features = self.final_conv(combined_features)
        combined_features = self.final_activation(combined_features)

        return unet_output, combined_features
