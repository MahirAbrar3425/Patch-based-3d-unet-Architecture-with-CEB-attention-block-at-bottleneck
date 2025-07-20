import torch
import torch.nn as nn
from modifiedTinySwin_ import TinySwinTransformer  # Make sure this handles 3D inputs
from td_attention_unet import UNet3D
class UNet3DWithSkipSwinPurifier(nn.Module):
    def __init__(self, in_channels, out_channels, f_maps=16, final_sigmoid=False,
                 layer_order='crg', num_groups=8, swin_embed_dim=96):
        super().__init__()

        # Base UNet3D model
        #from UNet3D import UNet3D  # import your UNet3D class
        self.unet_core = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups
        )

        # Feature map sizes per level
        if isinstance(f_maps, int):
            f_maps = [f_maps * 2**k for k in range(4)]

        # Create skip purifiers (no purifier for the bottleneck)
        self.skip_purifiers = nn.ModuleList([
            TinySwinTransformer(input_dim=ch, embed_dim=swin_embed_dim)
            for ch in reversed(f_maps[:-1])  # skip last (bottleneck) level
        ])

        self.final_activation = self.unet_core.final_activation
        self.final_conv = self.unet_core.final_conv

    def forward(self, x):
        enc_feats = []

        # Encoder with feature collection
        for encoder in self.unet_core.encoders:
            x = encoder(x)
            enc_feats.insert(0, x)  # Reverse order for decoder

        skip_feats = enc_feats[1:]  # Skip features (exclude bottleneck)
        x = enc_feats[0]            # Bottleneck feature

        # Decode with purified skip connections
        for decoder, skip_feat, purifier in zip(self.unet_core.decoders, skip_feats, self.skip_purifiers):
            purified_skip = purifier(skip_feat)  # Purified skip connection
            x = decoder(purified_skip, x)

        x = self.final_conv(x)
        if not self.training:
            x = self.final_activation(x)
        return x
