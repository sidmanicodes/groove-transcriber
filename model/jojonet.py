# Imports
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_layers import FreqBN, AttentionBlock

class JojoNet(nn.Module):
    """A U-Net model with attention used for converting drum grooves to tensors"""
    def __init__(self) -> None:
        super().__init__()
        self.freqbn = FreqBN(num_freq_bins=513)

        # --------------
        # Encoder layers
        # --------------
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=256),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=512),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----------
        # Bottleneck
        # ----------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=1024)
        )

        # -----------------------------
        # Decoder layers with attention
        # -----------------------------
        self.upconv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.attn4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=512),
            nn.Dropout2d()
        )

        self.upconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1)
        self.attn3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d()
        )

        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1)
        self.attn2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.Dropout2d()
        )
        
        self.upconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1)
        self.attn1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout2d()
        )

        # Final convolution for output
        self.final = nn.Conv2d(in_channels=64, out_channels=11, kernel_size=1)

    def forward(self, x: torch.Tensor, training: bool = True, desired_ticks: Optional[int] = None) -> torch.Tensor:
        # print(x.shape)
        # --------------
        # Encoder layers
        # --------------
        enc1 = self.pool1(self.encoder1(x))
        # print(f"enc1: {enc1.shape}")
        enc2 = self.pool2(self.encoder2(enc1))
        # print(f"enc2: {enc2.shape}")
        enc3 = self.pool3(self.encoder3(enc2))
        # print(f"enc3: {enc3.shape}")
        enc4 = self.pool4(self.encoder4(enc3))
        # print(f"enc4: {enc4.shape}")
        
        # ----------
        # Bottleneck
        # ----------
        bottleneck = self.bottleneck(enc4)
        # print(f"bottleneck: {bottleneck.shape}")

        # --------------
        # Decoder layers
        # --------------
        # print("----------------Decoder layer 4----------------")
        dec4 = self.upconv4(bottleneck)
        attn4 = self.attn4(g=dec4, x=enc4)
        # print(f"Decoder shape: {dec4.shape} | Attention shape {attn4.shape}")
        attn4 = F.interpolate(input=attn4, size=dec4.shape[2:], mode='bilinear', align_corners=True)
        dec4 = self.decoder4(torch.concat((dec4, attn4), dim=1))
        # print(f"dec4: {dec4.shape}")
        
        # print("----------------Decoder layer 3----------------")
        dec3 = self.upconv3(dec4)
        attn3 = self.attn3(g=dec3, x=enc3)
        attn3 = F.interpolate(input=attn3, size=dec3.shape[2:], mode='bilinear', align_corners=True)
        # print(f"Decoder shape: {dec3.shape} | Attention shape {attn3.shape}")
        dec3 = self.decoder3(torch.concat((dec3, attn3), dim=1))
        # print(f"dec3: {dec3.shape}")
        
        # print("----------------Decoder layer 2----------------")
        dec2 = self.upconv2(dec3)
        attn2 = self.attn2(g=dec2, x=enc2)
        # print(f"Decoder shape: {dec2.shape} | Attention shape {attn2.shape}")
        attn2 = F.interpolate(input=attn2, size=dec2.shape[2:], mode='bilinear', align_corners=True)
        dec2 = self.decoder2(torch.concat((dec2, attn2), dim=1))
        # print(f"dec2: {dec2.shape}")
        
        # print("----------------Decoder layer 1----------------")
        dec1 = self.upconv1(dec2)
        attn1 = self.attn1(g=dec1, x=enc1)
        # print(f"Decoder shape: {dec1.shape} | Attention shape {attn1.shape}")
        attn1 = F.interpolate(input=attn1, size=dec1.shape[2:], mode='bilinear', align_corners=True)
        dec1 = self.decoder1(torch.concat((dec1, attn1), dim=1))

        output = self.final(dec1)
        batch_size, num_channels, height, width = output.shape

        # Flatten the output
        output = output.view(batch_size, num_channels, height*width)


        # If in training mode, apply interpolation to get to scale to the desired number of ticks
        # This is done in order to precisely match the dimensions of the corresponding midi tensor (the label)
        # This step is not necessary during inference since we will have a variable signal length
        # if training and desired_ticks:
        # output = F.interpolate(
        #     input=output, 
        #     size=5760, 
        #     mode='bilinear', 
        #     align_corners=True
        # )

        return output