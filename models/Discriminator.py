import torch
import torch.nn as nn
import torch.nn.parallel


def Conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=1, padding=1, bias=False)


def Block3x3_LeakyReLU(in_channels, out_channels):
    return nn.Sequential(
        Conv3x3(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )


def DownBlock(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )


def encode_x16(ndf):
    return nn.Sequential(
        # 3 x w x h -> ndf x w/2 x h/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        # ndf x w/2 x h/2 -> ndf*2 x w/4 x h/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),

        # ndf*2 x w x h -> ndf*4 x w/4 x h/4
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),

        # ndf*4 x w x h -> ndf*8 x w/4 x h/4
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),
    )


class DiscriminatorGetLogits(nn.Module):
    def __init__(self, ndf, nef, b_condition=False):
        super(DiscriminatorGetLogits, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.b_condition = b_condition
        if self.b_condition:
            self.jointConv = Block3x3_LeakyReLU(ndf * 8 + nef, ndf * 8)

        self.output_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid()
        )

    def forward(self, h_code, c_code=None):
        if self.b_condition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)

            hc_code = torch.cat((h_code, c_code), dim=1)
            hc_code = self.jointConv(hc_code)
        else:
            hc_code = h_code

        return self.output_logits(hc_code).view(-1)


class DiscriminatorNetwork(nn.Module):
    def __init__(self, b_jcu=True, down_sample_count=0):
        super(DiscriminatorNetwork, self).__init__()

        ndf = 64
        nef = 256

        self.img_code = encode_x16(ndf)
        self.sample_list = nn.ModuleList()

        assert down_sample_count <= 2, 'bro! u try to crash the server? UNSUPPORTED'

        for down_sample in range(1, down_sample_count + 1):
            self.sample_list.append(DownBlock(ndf * 8 * down_sample,
                                              ndf * 16 * down_sample))

        for down_sample in range(down_sample_count, 0, -1):
            self.sample_list.append(Block3x3_LeakyReLU(ndf * 16 * down_sample,
                                                       ndf * 8 * down_sample))

        if b_jcu:
            self.unconditional_discriminator = DiscriminatorGetLogits(ndf, nef, b_condition=False)
        else:
            self.unconditional_discriminator = None
        self.conditional_discriminator = DiscriminatorGetLogits(ndf, nef, b_condition=True)

    def forward(self, x_var):
        x_code = self.img_code(x_var)
        for module in self.sample_list:
            x_code = module(x_code)

        return x_code
