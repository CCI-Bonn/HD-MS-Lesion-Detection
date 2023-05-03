"""                                                                                                            
:AUTHOR: Hagen Meredig                                                                                         
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital                                    
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de                                                                  
:SINCE: August 18, 2021
"""

import torch
from torch import Tensor
import torch.nn as nn


class ConvAutoencoder2D(nn.Module):
    def __init__(self, n_channels=1, channel_blocks=[16, 32, 64]):
        super(ConvAutoencoder2D, self).__init__()
        lblock = len(channel_blocks)

        encoder_list = []
        for i in range(len(channel_blocks)):
            if i == 0:
                encoder_list.append(
                    nn.Conv2d(n_channels, channel_blocks[i], 3, stride=2, padding=1)
                )
            else:
                if i < lblock - 1:
                    encoder_list.append(
                        nn.Conv2d(
                            channel_blocks[i - 1],
                            channel_blocks[i],
                            3,
                            stride=2,
                            padding=1,
                        )
                    )
                else:
                    encoder_list.append(
                        nn.Conv2d(channel_blocks[i - 1], channel_blocks[i], 7)
                    )
            if i < lblock - 1:
                encoder_list.append(nn.ReLU())

        decoder_list = []
        for i in range(len(channel_blocks)):
            if i == 0:
                decoder_list.append(
                    nn.ConvTranspose2d(
                        channel_blocks[lblock - 1], channel_blocks[lblock - 2], 7
                    )
                )
            else:
                if i < lblock - 1:
                    decoder_list.append(
                        nn.ConvTranspose2d(
                            channel_blocks[lblock - 1 - i],
                            channel_blocks[lblock - 2 - i],
                            3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                        )
                    )
                else:
                    decoder_list.append(
                        nn.ConvTranspose2d(
                            channel_blocks[lblock - 1 - i],
                            n_channels,
                            3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                        )
                    )
            if i < lblock - 1:
                decoder_list.append(nn.ReLU())
            else:
                decoder_list.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_list)
        self.decoder = nn.Sequential(*decoder_list)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
