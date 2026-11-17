# import the necessary packages
from . import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
import torch

KERNEL_SIZE = 3

        
class Block(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(in_channels, out_channels, KERNEL_SIZE, padding=1)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, KERNEL_SIZE, padding=1)

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))


class Hourglass(Module):
    def __init__(self, enc_channels=(3, 16, 32, 64), dec_channels=(64, 32, 16, 8)):

        super().__init__()

        # store the encoder blocks and maxpooling layer
        self.enc_blocks = ModuleList([Block(enc_channels[i], enc_channels[i+1])
                                     for i in range(len(enc_channels) - 1)])
        # pooling of 2 * 2 
        self.pooling = MaxPool2d(kernel_size = 2, stride=2)

        # initialize decoder # channels, upsampler blocks, and decoder blocks
        self.channels = dec_channels
        self.dec_blocks = ModuleList(
            [Block(dec_channels[i], dec_channels[i + 1])
             for i in range(len(dec_channels) - 1)])
        
        # TODO: upconvolutions 
        self.upconv = ModuleList([ConvTranspose2d(dec_channels[i], dec_channels[i + 1], kernel_size=2, stride=2)
             for i in range(len(dec_channels) - 1)])

        # the last conv :
        self.last_conv = Conv2d(self.channels[-1], 1, 3, padding=1)
    
    def forward(self, x):
        # loop through the encoder blocks
        for block in self.enc_blocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)

            # pooling
            x = self.pooling(x)

        # decoder: loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.dec_blocks[i](x)

            # upconvolutions
            x = self.upconv[i](x)
        #
        # return the final decoder output
        return self.last_conv(x)


    
