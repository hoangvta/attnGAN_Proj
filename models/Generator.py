import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

import config.settings as config

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        channels = x.size(1)
        assert channels % 2 == 0, "odd channels"
        channels = channels // 2
        return x[:, :channels] * F.sigmoid(x[:, channels:])


def Conv1x1(in_channels, out_channels, bias=False):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=1, padding=0, bias=bias)


def Conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=1, padding=1, bias=False)


def UpBlock(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        Conv3x3(in_channels, out_channels * 2),
        nn.BatchNorm2d(out_channels * 2),
        GLU()
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            Conv3x3(channels, channels * 2),
            nn.BatchNorm2d(channels * 2),
            GLU(),
            Conv3x3(channels, channels),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class CA_Net(nn.Module):
    def __init__(self):
        super(CA_Net, self).__init__()
        self.t_dim = config.EMBEDDING_DIM
        self.c_dim = config.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        log_var = x[:, self.c_dim:]
        return mu, log_var

    def parameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(config.DEVICE)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, log_var = self.encode(text_embedding)
        c_code = self.parameterize(mu, log_var)
        return c_code, mu, log_var


class AttentionModule(nn.Module):
    def __init__(self, idf, cdf):
        super(AttentionModule, self).__init__()
        self.convContext = Conv1x1(cdf, idf)
        self.softmax = nn.Softmax(dim=1)
        self.mask = None

    def setMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        input_height, input_width = input.size(2), input.size(3)
        pixel_count = input_height * input_width
        batch_size, context_count = context.size(0), context.size(2)

        target = input.view(batch_size, -1, pixel_count)
        targetT = torch.transpose(target, 1, 2).contiguous()

        sourceT = context.unsqueeze(3)
        sourceT = self.convContext(sourceT).squeeze(3)

        attn = torch.bmm(targetT, sourceT).view(batch_size*pixel_count, context_count)
        if self.mask is not None:
            mask = self.mask.repeat(pixel_count, 1).bool()
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.softmax(attn).view(batch_size, pixel_count, context_count)
        attn = torch.transpose(attn, 1, 2).contiguous()

        weight = torch.bmm(sourceT, attn).view(batch_size, -1, input_height, input_width)
        attn = attn.view(batch_size, -1, input_height, input_width)

        return weight, attn


class InitStageGenerator(nn.Module):
    def __init__(self, ngf, ncf):
        super(InitStageGenerator, self).__init__()
        self.gf_dim = ngf
        self.in_dim = config.Z_DIM + ncf

        self.define_modules()

    def define_modules(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU()
        )

        self.upsample1 = UpBlock(ngf, ngf // 2)
        self.upsample2 = UpBlock(ngf // 2, ngf // 4)
        self.upsample3 = UpBlock(ngf // 4, ngf // 8)
        self.upsample4 = UpBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code):
        cz_code = torch.cat((c_code, z_code), dim=1)
        output_code = self.fc(cz_code)
        output_code = output_code.view(-1, self.gf_dim, 4, 4)  # ngf x 4 x 4
        output_code = self.upsample1(output_code)  # ngf / 2 x 8 x 8
        output_code = self.upsample2(output_code)  # ngf / 4 x 16 x 16
        output_code = self.upsample3(output_code)  # ngf / 8 x 32 x 32
        output_code = self.upsample4(output_code)  # ngf / 16 x 64 x 64

        return output_code


class NextStageGenerator(nn.Module):
    def __init__(self, ngf, nef, ncf):
        super(NextStageGenerator, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = 2
        self.define_modules()

    def _make_layer(self, block, channels):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def define_modules(self):
        self.attn = AttentionModule(self.gf_dim, self.ef_dim)
        self.residual = self._make_layer(ResidualBlock, self.gf_dim * 2)
        self.upSample = UpBlock(self.gf_dim * 2, self.gf_dim)

    def forward(self, h_code, c_code, word_emb, mask):
        self.attn.setMask(mask)
        c_code, attn = self.attn(h_code, word_emb)
        hc_code = torch.cat((h_code, c_code), dim=1)
        output = self.residual(hc_code)

        output = self.upSample(output)

        return output, attn


class GetGenerateImage(nn.Module):
    def __init__(self, ngf):
        super(GetGenerateImage, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            Conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        return self.img(h_code)


class GenerativeNetwork(nn.Module):

    def __init__(self):
        super(GenerativeNetwork, self).__init__()
        ngf = config.GF_DIM
        nef = config.EMBEDDING_DIM
        ncf = config.CONDITION_DIM

        self.ca_net = CA_Net()

        self.h_net1 = InitStageGenerator(ngf * 16, ncf)
        self.img_net1 = GetGenerateImage(ngf)

        self.h_net2 = NextStageGenerator(ngf, nef, ncf)
        self.img_net2 = GetGenerateImage(ngf)

    def forward(self, z_code, sent_emb, word_emb, mask):
        """
        summary(model, [(1, 100), # batch_size x ngf
                        (1, 256), # batch_size x nef
                        (1, 256, 100), # batch_size x nef x ngf
                        (1, 100)]) # batch_size x ngf
        """
        fake_images = []
        att_map = []

        c_code, mu, log_var = self.ca_net(sent_emb)
        h_code1 = self.h_net1(z_code, c_code)
        fake_img1 = self.img_net1(h_code1)
        fake_images.append(fake_img1)

        h_code2, attn2 = self.h_net2(h_code1, c_code, word_emb, mask)
        fake_img2 = self.img_net2(h_code2)
        fake_images.append(fake_img2)

        if attn2 is not None:
            att_map.append(attn2)

        return fake_images, att_map, mu, log_var
