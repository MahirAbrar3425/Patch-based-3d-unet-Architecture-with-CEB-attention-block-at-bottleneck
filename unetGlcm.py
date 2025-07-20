import torch
import torch.nn as nn
import torch.nn.functional as F

from BuildingBlocks import Encoder, Decoder, FinalConv, DoubleConv, ExtResNetBlock, SingleConv

# --- Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        se = self.pool(x).view(B, C)
        se = self.fc(se).view(B, C, 1, 1, 1)
        return x * se

# --- UNet3D with GLCM + SE ---
def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]

class UNet3D(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 final_sigmoid, 
                 f_maps=16, 
                 layer_order='crg', 
                 num_groups=8,
                 glcm_channels=4,
                 use_glcm=True,     # <--- added comma properly
                 use_se=True,       # <--- added comma properly
                 **kwargs):
        super(UNet3D, self).__init__()

        # Save options inside the model
        self.use_glcm = use_glcm
        self.use_se = use_se

        if isinstance(f_maps, int):
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(
                    in_channels + (glcm_channels if self.use_glcm else 0), 
                    out_feature_num, 
                    apply_pooling=False,
                    basic_module=DoubleConv, 
                    conv_layer_order=layer_order, 
                    num_groups=num_groups
                )
            else:
                encoder = Encoder(
                    f_maps[i - 1], 
                    out_feature_num, 
                    basic_module=DoubleConv,
                    conv_layer_order=layer_order, 
                    num_groups=num_groups
                )
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(
                in_feature_num, 
                out_feature_num, 
                basic_module=DoubleConv,
                conv_layer_order=layer_order, 
                num_groups=num_groups
            )
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

        # GLCM path (only if use_glcm)
        if self.use_glcm:
            self.glcm_conv = nn.Sequential(
                nn.Conv3d(in_channels, glcm_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

            if self.use_se:
                self.se_block = SEBlock(in_channels + glcm_channels)

            self.glcm_skip = nn.Sequential(
                nn.Conv3d(glcm_channels, f_maps[0], kernel_size=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        B, C, D, H, W = x.shape

        if self.use_glcm:
            # Step 1: GLCM feature extraction
            glcm_features = self.glcm_conv(x)  # (B, glcm_channels, D, H, W)

            # Step 2: Concatenate input + GLCM
            x = torch.cat([x, glcm_features], dim=1)  # (B, C + glcm_channels, D, H, W)

            if self.use_se:
                # Step 3: Squeeze-and-Excitation after fusion
                x = self.se_block(x)

        # --- UNet3D Encoder path ---
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)

        # Save early encoder feature map (before decoding) for GLCM shortcut
        x_before_decoder = encoders_features[0]

        pool_fea = self.avg_pool(x_before_decoder).squeeze(0).squeeze(1).squeeze(1).squeeze(1)
        encoders_features = encoders_features[1:]

        # --- Decoder path ---
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)

        # --- Optional: Deep GLCM shortcut ---
        if self.use_glcm:
            glcm_deep = self.glcm_skip(glcm_features)  # (B, f_maps[0], D, H, W)
            x = x + glcm_deep  # residual add

        # --- Final prediction ---
        x = self.final_conv(x)

        if not self.training:
            x = self.final_activation(x)

        return x, pool_fea
